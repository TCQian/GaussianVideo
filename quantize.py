from vector_quantize_pytorch import VectorQuantize, ResidualVQ
import torch
from torch import nn
from utils import *
import torch.nn.functional as F
import constriction
import numpy as np

def grad_scale(x, scale):
    return (x - x * scale).detach() + x * scale

def ste(x):
    return (x.round() - x).detach() + x

class FakeQuantizationHalf(torch.autograd.Function):
    """performs fake quantization for half precision"""

    @staticmethod
    def forward(_, x):
        return x.half().float()

    @staticmethod
    def backward(_, grad_output):
        return grad_output

class UniformQuantizer(nn.Module):
    def __init__(self, signed=False, bits=8, learned=False, num_channels=1, entropy_type="none", weight=0.001):
        super().__init__()
        if signed:
            self.qmin = -2**(bits - 1)
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1

        self.learned = learned
        self.entropy_type = entropy_type
        if self.learned:
            self.scale = nn.Parameter(torch.ones(num_channels)/self.qmax, requires_grad=True)
            self.beta = nn.Parameter(torch.ones(num_channels)/self.qmax, requires_grad=True)

        self.weight = weight

    def _init_data(self, tensor):
        device = tensor.device
        t_min, t_max = tensor.min(dim=0)[0], tensor.max(dim=0)[0]
        scale = (t_max - t_min) / (self.qmax-self.qmin)
        self.beta.data = t_min.to(device)
        self.scale.data = scale.to(device)

    def forward(self, x):
        if self.learned:
            grad = 1.0 / ((self.qmax * x.numel()) ** 0.5)
            s_scale = grad_scale(self.scale, grad)
            beta_scale = grad_scale(self.beta, grad)
            code = ((x - beta_scale) / s_scale).clamp(self.qmin, self.qmax)
            quant = ste(code)
            dequant = quant * s_scale + beta_scale
        else:
            code = (x * self.qmax).clamp(self.qmin, self.qmax)
            quant = ste(code)
            dequant = quant / self.qmax

        bits, entropy_loss = 0, 0
        if not self.training:
            num_points, num_channels = x.shape
            bits = self.size(quant)
            # unit_bit = bits / num_points / num_channels
        return dequant, entropy_loss*self.weight, bits

    def size(self, quant):
        index_bits = 0
        compressed, histogram_table, unique = compress_matrix_flatten_categorical(quant.int().flatten().tolist())
        index_bits += get_np_size(compressed) * 8
        index_bits += get_np_size(histogram_table) * 8
        index_bits += get_np_size(unique) * 8 
        index_bits += self.scale.numel()*torch.finfo(self.scale.dtype).bits
        index_bits += self.beta.numel()*torch.finfo(self.beta.dtype).bits
        return index_bits

    def compress(self, x):
        code = ((x - self.beta) / self.scale).clamp(self.qmin, self.qmax)
        return code.round(), code.round()* self.scale + self.beta

    def decompress(self, x):
        return x * self.scale + self.beta
    
    def analyze(self, x, verbose=True):
        """
        Simple analysis of quantization: shows value distribution and bit usage.
        
        Args:
            x: Input tensor to analyze
            verbose: Print analysis results
        
        Returns:
            Dictionary with analysis metrics
        """
        quant, _ = self.compress(x)
        quant_np = quant.int().cpu().numpy().flatten()
        
        # Get entropy coding info
        compressed, histogram_table, unique = compress_matrix_flatten_categorical(quant_np.tolist())
        
        # Compute statistics
        unique_count = len(unique)
        total_values = len(quant_np)
        max_possible = self.qmax - self.qmin + 1
        
        # Value distribution
        value_counts = dict(zip(unique.tolist(), histogram_table.tolist()))
        
        # Entropy
        probs = histogram_table.astype(np.float64) / histogram_table.sum()
        entropy_bits = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Bit usage
        current_bits = total_values * np.ceil(np.log2(max_possible + 1))
        entropy_coded_bits = total_values * entropy_bits
        actual_bits = get_np_size(compressed) * 8 + get_np_size(histogram_table) * 8 + get_np_size(unique) * 8
        actual_bits += self.scale.numel() * torch.finfo(self.scale.dtype).bits
        actual_bits += self.beta.numel() * torch.finfo(self.beta.dtype).bits
        
        results = {
            'total_values': total_values,
            'unique_values': unique_count,
            'max_possible': max_possible,
            'utilization': unique_count / max_possible if max_possible > 0 else 0,
            'entropy_bits_per_value': entropy_bits,
            'current_bits': current_bits,
            'entropy_coded_bits': entropy_coded_bits,
            'actual_bits': actual_bits,
            'compression_ratio': current_bits / actual_bits if actual_bits > 0 else 1.0,
            'value_distribution': value_counts,
            'most_common': sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"UniformQuantizer Analysis (bits={self.qmax - self.qmin + 1})")
            print(f"{'='*60}")
            print(f"Total values: {total_values:,}")
            print(f"Unique values used: {unique_count} / {max_possible} ({results['utilization']*100:.1f}% utilization)")
            print(f"Entropy: {entropy_bits:.3f} bits/value")
            print(f"Current (fixed): {current_bits:,.0f} bits")
            print(f"Theoretical (entropy): {entropy_coded_bits:,.0f} bits")
            print(f"Actual (with overhead): {actual_bits:,.0f} bits")
            print(f"Compression ratio: {results['compression_ratio']:.2f}x")
            print(f"\nTop 10 most common values:")
            for val, count in results['most_common']:
                pct = count / total_values * 100
                print(f"  Value {val}: {count:,} times ({pct:.1f}%)")
        
        return results

class VectorQuantizer(nn.Module):
    def __init__(self, num_quantizers=1, codebook_dim=1, codebook_size=64, kmeans_iters=10, vector_type="vector"):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.vector_type = vector_type
        if self.num_quantizers == 1:
            if self.vector_type == "vector":
                self.quantizer = VectorQuantize(dim=codebook_dim, codebook_size=codebook_size, decay = 0.8, commitment_weight = 1., kmeans_init = True, 
                    kmeans_iters = kmeans_iters)#learnable_codebook=True, ema_update = False, orthogonal_reg_weight =1.)
        else:
            if self.vector_type == "vector":
                self.quantizer = ResidualVQ(dim=codebook_dim, codebook_size=codebook_size, num_quantizers=num_quantizers, decay = 0.8, commitment_weight = 1., kmeans_init = True, 
                    kmeans_iters = kmeans_iters) #learnable_codebook=True, ema_update=False, orthogonal_reg_weight=0., in_place_codebook_optimizer=torch.optim.Adam)

    def forward(self, x):
        if self.training:
            x, _, l_vq = self.quantizer(x)
            l_vq = torch.sum(l_vq)
            return x, l_vq, 0
        else:
            num_points, num_channels = x.shape
            x, embed_index, l_vq = self.quantizer(x)
            l_vq = torch.sum(l_vq)
            bits = self.size(embed_index)
            # unit_bit = bits / num_points / num_channels
            return x, l_vq, bits

    def size(self, embed_index):
        if self.num_quantizers == 1:
            if self.vector_type == "vector":
                codebook_bits = self.quantizer._codebook.embed.numel()*torch.finfo(self.quantizer._codebook.embed.dtype).bits
            elif self.vector_type == "ste":
                codebook_bits = self.quantizer.embedding.weight.data.numel()*torch.finfo(self.quantizer.embedding.weight.data.dtype).bits
            index_bits = 0
            compressed, histogram_table, unique = compress_matrix_flatten_categorical(embed_index.int().flatten().tolist())
            index_bits += get_np_size(compressed) * 8
            index_bits += get_np_size(histogram_table) * 8
            index_bits += get_np_size(unique) * 8  
        else:
            codebook_bits, index_bits = 0, 0
            for quantizer_index, layer in enumerate(self.quantizer.layers):
                if self.vector_type == "vector":
                    codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
                elif self.vector_type == "ste":
                    codebook_bits += layer.embedding.weight.data.numel()*torch.finfo(layer.embedding.weight.data.dtype).bits
            compressed, histogram_table, unique = compress_matrix_flatten_categorical(embed_index.int().flatten().tolist())
            index_bits += get_np_size(compressed) * 8
            index_bits += get_np_size(histogram_table) * 8
            index_bits += get_np_size(unique) * 8  
        total_bits = codebook_bits + index_bits
        #print("vq:", embed_index.shape, codebook_bits, index_bits)
        return total_bits

    def compress(self, x):
        x, embed_index, _ = self.quantizer(x)
        return x, embed_index

    def decompress(self, embed_index):
        recon = 0
        for i,layer in enumerate(self.quantizer.layers):
            recon += layer._codebook.embed[0, embed_index[:, i]]
        return recon
    
    def analyze(self, x, verbose=True):
        """
        Simple analysis of vector quantization: shows codebook usage and bit efficiency.
        
        Args:
            x: Input tensor to analyze
            verbose: Print analysis results
        
        Returns:
            Dictionary with analysis metrics
        """
        _, embed_index = self.compress(x)
        # embed_index should be a tensor with shape [num_points, num_quantizers]
        embed_index_np = embed_index.int().cpu().numpy()
        
        # Compute statistics
        total_vectors = embed_index_np.shape[0]
        num_quantizers = embed_index_np.shape[1]
        
        # Find unique index combinations (pairs/tuples)
        # Use numpy unique with axis=0 to find unique rows (combinations)
        unique_combinations, unique_inverse, unique_counts = np.unique(
            embed_index_np, return_inverse=True, return_counts=True, axis=0
        )
        unique_indices = len(unique_combinations)
        
        # For entropy coding, we still need to flatten for the compression function
        # But we'll track combinations separately for display
        compressed, histogram_table, unique = compress_matrix_flatten_categorical(embed_index_np.flatten().tolist())
        
        # Codebook size
        if self.num_quantizers == 1:
            codebook_size = self.quantizer._codebook.embed.shape[1]
        else:
            codebook_size = self.quantizer.layers[0]._codebook.embed.shape[1]
        
        max_possible_indices = codebook_size ** num_quantizers
        
        # Index distribution per quantizer
        index_distributions = []
        for q in range(num_quantizers):
            indices_q = embed_index_np[:, q]
            unique_q, counts_q = np.unique(indices_q, return_counts=True)
            index_distributions.append({
                'unique_count': len(unique_q),
                'codebook_size': codebook_size,
                'utilization': len(unique_q) / codebook_size,
                'distribution': dict(zip(unique_q.tolist(), counts_q.tolist()))
            })
        
        # Entropy
        probs = histogram_table.astype(np.float64) / histogram_table.sum()
        entropy_bits = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Bit usage
        current_bits_per_index = np.ceil(np.log2(codebook_size + 1))
        current_total = total_vectors * num_quantizers * current_bits_per_index
        entropy_coded_bits = total_vectors * num_quantizers * entropy_bits
        actual_bits = self.size(embed_index)
        
        results = {
            'total_vectors': total_vectors,
            'num_quantizers': num_quantizers,
            'codebook_size': codebook_size,
            'unique_index_combinations': unique_indices,
            'max_possible_combinations': max_possible_indices,
            'utilization': unique_indices / max_possible_indices if max_possible_indices > 0 else 0,
            'entropy_bits_per_index': entropy_bits,
            'current_bits': current_total,
            'entropy_coded_bits': entropy_coded_bits,
            'actual_bits': actual_bits,
            'compression_ratio': current_total / actual_bits if actual_bits > 0 else 1.0,
            'per_quantizer_stats': index_distributions,
            'most_common_indices': sorted(
                zip(unique_combinations.tolist(), unique_counts.tolist()),
                key=lambda x: x[1], reverse=True
            )[:10]
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"VectorQuantizer Analysis")
            print(f"{'='*60}")
            print(f"Total vectors: {total_vectors:,}")
            print(f"Quantizers: {num_quantizers}")
            print(f"Codebook size: {codebook_size}")
            print(f"Unique index combinations: {unique_indices} / {max_possible_indices} ({results['utilization']*100:.1f}% utilization)")
            print(f"Entropy: {entropy_bits:.3f} bits/index")
            print(f"Current (fixed): {current_total:,.0f} bits")
            print(f"Theoretical (entropy): {entropy_coded_bits:,.0f} bits")
            print(f"Actual (with overhead): {actual_bits:,.0f} bits")
            print(f"Compression ratio: {results['compression_ratio']:.2f}x")
            
            print(f"\nPer-quantizer codebook usage:")
            for q, stats in enumerate(index_distributions):
                print(f"  Quantizer {q}: {stats['unique_count']}/{stats['codebook_size']} entries used ({stats['utilization']*100:.1f}%)")
            
            print(f"\nTop 10 most common index combinations:")
            for idx_combo, count in results['most_common_indices']:
                pct = count / total_vectors * 100  # Percentage of vectors, not total indices
                # Format as [idx0, idx1, ...] for readability
                if isinstance(idx_combo, (list, tuple, np.ndarray)):
                    idx_str = "[" + ", ".join(map(str, idx_combo)) + "]"
                else:
                    idx_str = f"[{idx_combo}]"
                print(f"  {idx_str}: {count:,} times ({pct:.1f}%)")
        
        return results

def compress_matrix_flatten_categorical(matrix, return_table=False):
    '''
    :param matrix: np.array
    :return compressed, symtable
    '''
    matrix = np.array(matrix) #matrix.flatten()
    unique, unique_indices, unique_inverse, unique_counts = np.unique(matrix, return_index=True, return_inverse=True, return_counts=True, axis=None)
    min_value = np.min(unique)
    max_value = np.max(unique)
    unique = unique.astype(judege_type(min_value, max_value))
    message = unique_inverse.astype(np.int32)
    probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
    entropy_model = constriction.stream.model.Categorical(probabilities)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(message, entropy_model)
    compressed = encoder.get_compressed()
    return compressed, unique_counts, unique

def decompress_matrix_flatten_categorical(compressed, unique_counts, quant_symbol, symbol_length, symbol_shape):
    '''
    :param matrix: np.array
    :return compressed, symtable
    '''
    probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
    entropy_model = constriction.stream.model.Categorical(probabilities)
    decoder = constriction.stream.stack.AnsCoder(compressed)
    decoded = decoder.decode(entropy_model, symbol_length)
    decoded = quant_symbol[decoded].reshape(symbol_shape)#.astype(np.int32)
    return decoded


def judege_type(min, max):
    if min>=0:
        if max<=256:
            return np.uint8
        elif max<=65535:
            return np.uint16
        else:
            return np.uint32
    else:
        if max<128 and min>=-128:
            return np.int8
        elif max<32768 and min>=-32768:
            return np.int16
        else:
            return np.int32
        
def get_np_size(x):
    return x.size * x.itemsize
