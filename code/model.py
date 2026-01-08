# [file name]: model.py
import torch.nn.functional as F
from dgl.ops import edge_softmax
from utils import *
import torch
import math

class BayesianFeatureLearning(nn.Module):
    """Bayesian feature learning layer for microbes, genes, and diseases"""
    def __init__(self, in_size, hidden_size, num_components=3):
        super(BayesianFeatureLearning, self).__init__()
        self.hidden_size = hidden_size
        self.num_components = num_components
        
        # Bayesian parameters for each node type
        self.prior_mu = nn.ParameterDict({
            'g': nn.Parameter(torch.randn(num_components, hidden_size)),
            'm': nn.Parameter(torch.randn(num_components, hidden_size)),
            'd': nn.Parameter(torch.randn(num_components, hidden_size))
        })
        
        self.prior_sigma = nn.ParameterDict({
            'g': nn.Parameter(torch.ones(num_components, hidden_size)),
            'm': nn.Parameter(torch.ones(num_components, hidden_size)),
            'd': nn.Parameter(torch.ones(num_components, hidden_size))
        })
        
        # Posterior distribution networks
        self.posterior_networks = nn.ModuleDict({
            'g': nn.Sequential(
                nn.Linear(in_size['g'], hidden_size * 2),
                nn.CELU(alpha=3.0),
                nn.Linear(hidden_size * 2, num_components * hidden_size * 2)
            ),
            'm': nn.Sequential(
                nn.Linear(in_size['m'], hidden_size * 2),
                nn.CELU(alpha=3.0),
                nn.Linear(hidden_size * 2, num_components * hidden_size * 2)
            ),
            'd': nn.Sequential(
                nn.Linear(in_size['d'], hidden_size * 2),
                nn.CELU(alpha=3.0),
                nn.Linear(hidden_size * 2, num_components * hidden_size * 2)
            )
        })
        
        # Mixture weight networks
        self.mixture_weights = nn.ModuleDict({
            'g': nn.Sequential(
                nn.Linear(in_size['g'], num_components),
                nn.Softmax(dim=-1)
            ),
            'm': nn.Sequential(
                nn.Linear(in_size['m'], num_components),
                nn.Softmax(dim=-1)
            ),
            'd': nn.Sequential(
                nn.Linear(in_size['d'], num_components),
                nn.Softmax(dim=-1)
            )
        })
        
        # Feature transformation layers
        self.feature_transforms = nn.ModuleDict({
            'g': nn.Linear(in_size['g'], hidden_size),
            'm': nn.Linear(in_size['m'], hidden_size),
            'd': nn.Linear(in_size['d'], hidden_size)
        })
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for node_type in ['g', 'm', 'd']:
            # Initialize prior distributions
            nn.init.xavier_normal_(self.prior_mu[node_type])
            nn.init.ones_(self.prior_sigma[node_type])
            
            # Initialize network weights
            for layer in self.posterior_networks[node_type]:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=1.414)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
            
            for layer in self.mixture_weights[node_type]:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=1.414)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
            
            nn.init.xavier_normal_(self.feature_transforms[node_type].weight, gain=1.414)
            if self.feature_transforms[node_type].bias is not None:
                nn.init.constant_(self.feature_transforms[node_type].bias, 0)
    
    def reparameterize(self, mu, log_sigma):
        """Reparameterization trick for sampling"""
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def kl_divergence(self, posterior_mu, posterior_sigma, prior_mu, prior_sigma):
        """Calculate KL divergence between posterior and prior"""
        kl = 0.5 * (2 * torch.log(prior_sigma / posterior_sigma) - 1 + 
                    (posterior_sigma**2 + (posterior_mu - prior_mu)**2) / prior_sigma**2)
        return kl.sum(dim=-1)  # Sum along feature dimension
    
    def forward(self, features):
        """
        Args:
            features: Raw feature dictionary {'g': gene_feat, 'm': microbe_feat, 'd': disease_feat}
        Returns:
            bayesian_features: Bayesian feature dictionary
            kl_loss: KL divergence loss
        """
        bayesian_features = {}
        kl_loss = 0.0
        mixture_weights = {}
        
        for node_type in ['g', 'm', 'd']:
            # Ensure input features are on correct device
            feat = features[node_type]
            
            # Get mixture weights
            pi = self.mixture_weights[node_type](feat)  # [num_nodes, num_components]
            mixture_weights[node_type] = pi
            
            # Get posterior distribution parameters
            posterior_params = self.posterior_networks[node_type](feat)
            posterior_params = posterior_params.view(-1, self.num_components, self.hidden_size, 2)
            
            posterior_mu = posterior_params[:, :, :, 0]  # [num_nodes, num_components, hidden_size]
            posterior_log_sigma = posterior_params[:, :, :, 1]
            posterior_sigma = torch.exp(0.5 * posterior_log_sigma)
            
            # Reparameterization sampling
            samples = self.reparameterize(posterior_mu, posterior_log_sigma)  # [num_nodes, num_components, hidden_size]
            
            # Combine components using mixture weights
            weighted_samples = torch.sum(pi.unsqueeze(-1) * samples, dim=1)  # [num_nodes, hidden_size]
            
            # Add base feature transformation
            base_features = self.feature_transforms[node_type](feat)
            bayesian_features[node_type] = base_features + weighted_samples
            
            # Calculate KL divergence for each component
            for k in range(self.num_components):
                prior_mu_k = self.prior_mu[node_type][k].unsqueeze(0).expand_as(posterior_mu[:, k, :])
                prior_sigma_k = torch.exp(0.5 * self.prior_sigma[node_type][k]).unsqueeze(0).expand_as(posterior_sigma[:, k, :])
                
                component_kl = self.kl_divergence(
                    posterior_mu[:, k, :], posterior_sigma[:, k, :],
                    prior_mu_k, prior_sigma_k
                )
                # Weight KL divergence by mixture weights
                kl_loss += torch.mean(pi[:, k] * component_kl)
        
        return bayesian_features, kl_loss, mixture_weights

class FourierPositionEmbedding(nn.Module):
    """Fourier position embedding layer"""
    def __init__(self, hidden_size, max_position=500):
        super(FourierPositionEmbedding, self).__init__()
        self.hidden_size = hidden_size
        
        # Position embeddings for each node type
        self.position_embeddings = nn.ModuleDict({
            'g': self._create_fourier_embedding(max_position),
            'm': self._create_fourier_embedding(max_position), 
            'd': self._create_fourier_embedding(max_position)
        })
        
        # Learnable scaling parameters
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        
    def _create_fourier_embedding(self, max_position):
        """Create Fourier position embedding matrix"""
        position = torch.arange(0, max_position).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2) * 
                           -(math.log(10000.0) / self.hidden_size))
        
        pe = torch.zeros(max_position, self.hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Embedding.from_pretrained(pe, freeze=False)
    
    def forward(self, bayesian_features, node_type, node_indices):
        """
        Args:
            bayesian_features: Bayesian features [num_nodes, hidden_size]
            node_type: Node type to select position embedding
            node_indices: Node indices
        """
        pos_embedding = self.position_embeddings[node_type](node_indices)
        
        # Combine features with position embeddings
        enhanced_features = (self.alpha * bayesian_features + 
                           self.beta * pos_embedding)
        
        return enhanced_features

class CMRL(nn.Module):
    """Causal Multi-relational Learning model with Bayesian feature learning"""
    def __init__(self, device, meta_paths, test_data, in_size, hidden_size, num_heads, dropout, etypes):
        super(CMRL, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Bayesian feature learning layer
        self.bayesian_learning = BayesianFeatureLearning(in_size, hidden_size)
        
        # Fourier position encoding
        self.fourier_embedding = FourierPositionEmbedding(hidden_size)
        
        # Relation vectors and subsequent layers
        r_vec = nn.Parameter(torch.empty(size=(3, self.hidden_size // 2, 2)))
        nn.init.xavier_normal_(r_vec, gain=1.414)
        
        # Heterogeneous graph neural network layer
        self.layers1 = HCMGNN_Layer(self.device, meta_paths, test_data, hidden_size, r_vec, num_heads, dropout, etypes,
                                    name=['g', 'm', 'd'])
        
        # Information bottleneck layer
        self.ib = InformationBottleneck(dropout)
        
        # MLP decoder
        self.decoder = MLPDecoder(dropout, self.hidden_size, self.num_heads, 1)
        
        # Move model to device
        self.to(device)
        
    def get_embed_map(self, features, embed_features, data):
        """Get embedding map for nodes"""
        stack_embedding = {'g': list(), 'm': list(), 'd': list()}
        
        for i, node_type in enumerate(['g', 'm', 'd']):
            for j in range(len(data)):
                node_id = data[j, i].item()
                if node_id < len(embed_features[node_type]):
                    embedding = embed_features[node_type][node_id]
                else:
                    # If node ID is out of range, use original features
                    embedding = torch.hstack([features[node_type][node_id]] * self.num_heads)
                
                stack_embedding[node_type].append(embedding)
            
            stack_embedding[node_type] = torch.stack(stack_embedding[node_type], dim=0)
        
        # Concatenate embeddings from all node types
        embedding_concat = torch.cat([stack_embedding['g'], stack_embedding['m'], stack_embedding['d']], dim=1)
        return embedding_concat

    def forward(self, g, inputs, data):
        """Forward pass"""
        # Ensure input data is on correct device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        data = torch.tensor(data, device=self.device, dtype=torch.long)
        
        # Bayesian feature learning
        bayesian_features, kl_loss, mixture_weights = self.bayesian_learning(inputs)
        
        # Apply Fourier position encoding to Bayesian features
        h_trans = {}
        g_indices = torch.arange(inputs['g'].shape[0], device=self.device)
        m_indices = torch.arange(inputs['m'].shape[0], device=self.device) 
        d_indices = torch.arange(inputs['d'].shape[0], device=self.device)
        
        h_trans['g'] = self.fourier_embedding(nn.CELU(alpha=3.0)(bayesian_features['g']), 'g', g_indices)
        h_trans['m'] = self.fourier_embedding(nn.CELU(alpha=3.0)(bayesian_features['m']), 'm', m_indices)
        h_trans['d'] = self.fourier_embedding(nn.CELU(alpha=3.0)(bayesian_features['d']), 'd', d_indices)

        # Subsequent processing through GNN layers
        h_trans_embed = self.layers1(g.to(self.device), h_trans)
        h_concat = self.get_embed_map(h_trans, h_trans_embed, data)
        
        # Apply information bottleneck
        jointF, vec_mean, vec_cov = self.ib(h_concat)
        
        # Decode to get prediction probability
        prob = self.decoder(jointF)
        
        return prob, vec_mean, vec_cov, kl_loss, mixture_weights

class MessageAggregator(nn.Module):
    """Message aggregator for meta-path instances"""
    def __init__(self, device, num_heads, hidden_size, attn_drop, alpha, name):
        super(MessageAggregator, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.celu = nn.CELU(alpha=3.0)
        self.softmax = edge_softmax
        self.device = device
        
        # Attention dropout
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
        
        # Attention layers
        self.attn1 = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
        
        self.attn2 = nn.Parameter(torch.empty(size=(1, self.num_heads, self.hidden_size)))
        nn.init.xavier_normal_(self.attn2.data, gain=1.414)
        
        self.name = name

    def forward(self, nodes, metapath_instances, metapath_embedding, features):
        """Aggregate messages from meta-path instances"""
        h_ = []
        for i in range(len(nodes)):
            # Find meta-path instances containing this node
            index = metapath_instances[metapath_instances[self.name] == nodes[i]].index.tolist()
            
            if index:
                # Get embeddings for meta-path instances
                node_metapath_embedding = metapath_embedding[index]  # (E, hidden_size)
                node_metapath_embedding = torch.cat([node_metapath_embedding] * self.num_heads, dim=1)  # (E, hidden_size*num_heads)
                node_metapath_embedding = node_metapath_embedding.unsqueeze(dim=0)  # (1, E, hidden_size*num_heads)
                eft = node_metapath_embedding.permute(1, 0, 2).view(-1, self.num_heads, self.hidden_size)  # (E, num_heads, hidden_size)
                eft = self.celu(eft)
                
                # Get node features
                node_embedding = torch.vstack([features[i]] * len(index))  # (E, hidden_size)
                
                # Calculate attention scores
                a1 = self.celu(self.attn1(node_embedding))
                a2 = (eft * self.attn2).sum(dim=-1)
                a = (a1 + a2).unsqueeze(dim=-1)
                a = self.celu(a)
                attention = F.softmax(a, dim=0)
                attention = self.attn_drop(attention)
                
                # Aggregate weighted embeddings
                h = F.celu((attention * eft).sum(dim=0), alpha=3.0).view(-1, self.hidden_size * self.num_heads)
                h_.append(h[0].to(self.device))
            else:
                # If no meta-path instances, use zero embedding
                node_embedding = torch.zeros(self.hidden_size * self.num_heads)
                h_.append(node_embedding.to(self.device))
        
        return torch.stack(h_, dim=0)

class Subgraph_Fusion(nn.Module):
    """Fuse embeddings from different subgraphs"""
    def __init__(self, in_size, hidden_size=128):
        super(Subgraph_Fusion, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size*2),
            nn.CELU(alpha=3.0),
            nn.Linear(hidden_size*2, hidden_size),
            nn.CELU(alpha=3.0),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def weights_init(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=1.414)

    def forward(self, z):
        """Fuse multiple subgraph embeddings"""
        # Calculate attention weights
        w = self.project(z).mean(0)
        beta_ = torch.softmax(w, dim=0)
        
        # Expand weights and apply to embeddings
        beta = beta_.expand((z.shape[0],) + beta_.shape)
        return (beta * z).sum(1), beta_

class SemanticEncoder(nn.Module):
    """Encode semantic information from meta-paths"""
    def __init__(self, device, layer_num_heads, hidden_size, r_vec, etypes):
        super(SemanticEncoder, self).__init__()
        self.device = device
        self.num_heads = layer_num_heads
        self.hidden_size = hidden_size
        self.r_vec = r_vec
        self.etypes = etypes

    def forward(self, edata):
        """Encode edge data along meta-paths"""
        # Reshape edge data
        edata = edata.reshape(edata.shape[0], edata.shape[1], edata.shape[2] // 2, 2)
        final_r_vec = torch.zeros([edata.shape[1], self.hidden_size // 2, 2]).to(self.device)
        
        # Normalize relation vectors
        r_vec = F.normalize(self.r_vec, p=2, dim=2)
        r_vec = torch.stack((r_vec, r_vec), dim=1)
        r_vec[:, 1, :, 1] = -r_vec[:, 1, :, 1]
        r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1], 2)
        
        # Initialize final relation vector
        final_r_vec[-1, :, 0] = 1
        
        # Propagate relation vectors through meta-path
        for i in range(final_r_vec.shape[0] - 2, -1, -1):
            if self.etypes[i] is not None:
                final_r_vec[i, :, 0] = (final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 0] - 
                                       final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 1])
                final_r_vec[i, :, 1] = (final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 1] + 
                                       final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 0])
            else:
                final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone()
                final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 1].clone()
        
        # Apply relation transformations to edge data
        for i in range(edata.shape[1] - 1):
            temp1 = (edata[:, i, :, 0].clone() * final_r_vec[i, :, 0] - 
                    edata[:, i, :, 1].clone() * final_r_vec[i, :, 1])
            temp2 = (edata[:, i, :, 0].clone() * final_r_vec[i, :, 1] + 
                    edata[:, i, :, 1].clone() * final_r_vec[i, :, 0])
            edata[:, i, :, 0] = temp1
            edata[:, i, :, 1] = temp2
        
        # Reshape and average to get meta-path embedding
        edata = edata.reshape(edata.shape[0], edata.shape[1], -1)
        metapath_embedding = torch.mean(edata, dim=1)
        
        return metapath_embedding

class CMRL_Layer(nn.Module):
    """CMRL layer for heterogeneous graph learning"""
    def __init__(self, device, meta_paths, test_data, hidden_size, r_vec, layer_num_heads, dropout, etypes, name):
        super(CMRL_Layer, self).__init__()
        self.device = device
        self.num_heads = layer_num_heads
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.r_vec = r_vec
        self.etypes = etypes
        self.hidden_size = hidden_size
        self.test_data = test_data
        
        # Semantic encoders for each meta-path
        self.semantic_encoder_layer = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.semantic_encoder_layer.append(
                SemanticEncoder(self.device, self.num_heads, self.hidden_size, self.r_vec, self.etypes[i]))
        
        # Message aggregators for each node type
        self.message_aggregator_layer = nn.ModuleList()
        for i in name:
            self.message_aggregator_layer.append(
                MessageAggregator(self.device, self.num_heads, self.hidden_size, attn_drop=dropout, alpha=0.01, name=i))
        
        # Subgraph fusion layer
        self.subgraph_fusion = Subgraph_Fusion(in_size=self.hidden_size * self.num_heads)
        
        # Helper modules
        self.separate_metapath_subgraph = Separate_subgraph()
        self.exclude_test = Prevent_leakage(self.test_data)

    def stack_embedding(self, embeddings):
        """Stack embeddings from different subgraphs"""
        subgraph_num_nodes = [embeddings[i].size()[0] for i in range(len(embeddings))]
        
        if subgraph_num_nodes.count(subgraph_num_nodes[0]) == len(subgraph_num_nodes):
            # All subgraphs have same number of nodes
            embeddings = torch.stack(embeddings, dim=1)
        else:
            # Pad subgraphs with zeros to make them equal size
            for i in range(0, len(embeddings)):
                index = max(subgraph_num_nodes) - subgraph_num_nodes[i]
                if index != 0:
                    h_ = torch.zeros(index, self.hidden_size * self.num_heads)
                    embeddings[i] = torch.cat((embeddings[i], h_), dim=0)
            embeddings = torch.stack(embeddings, dim=1)
        
        return embeddings

    def generate_metapath_instances(self, g, meta_path):
        """Generate meta-path instances from graph"""
        edges = [g.edges(etype=f"{meta_path[j]}_{meta_path[j + 1]}") for j in range(len(meta_path) - 1)]
        edges = [[edges[i][j].tolist() for j in range(len(edges[i]))] for i in range(len(edges))]
        
        # Create dataframes for edges
        df_0 = pd.DataFrame(edges[0], index=list(meta_path)[:2]).T
        df_1 = pd.DataFrame(edges[1], index=list(meta_path)[-2:]).T
        
        # Merge to find complete meta-path instances
        metapath_instances = pd.merge(df_0, df_1, how='inner')
        filt_metapath_instances = metapath_instances[['g', 'm', 'd']]
        
        # Exclude test data to prevent information leakage
        filt_metapath_instances = self.exclude_test(filt_metapath_instances)
        metapath_instances = filt_metapath_instances[list(meta_path)]
        
        return metapath_instances

    def forward(self, g, h):
        """Forward pass through CMRL layer"""
        # Cache graph computations
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = self.separate_metapath_subgraph(g, meta_path)
        
        semantic_embeddings = {'g': [], 'm': [], 'd': []}
        nodes_embeddings = {}
        
        # Process each meta-path
        for i, meta_path in enumerate(self.meta_paths):
            edata_list = []
            new_g = self._cached_coalesced_graph[meta_path]
            
            # Generate meta-path instances
            metapath_instances = self.generate_metapath_instances(new_g, meta_path)
            
            # Collect edge data along meta-path
            for j in range(len(meta_path)):
                edata_list.append(
                    F.embedding(
                        torch.tensor(metapath_instances.iloc[:, j]).to(self.device), 
                        h[list(meta_path)[j]].to(self.device)
                    ).unsqueeze(1))
            
            edata = torch.hstack(edata_list)
            
            # Encode semantic information
            metapathembedding = (nn.CELU(alpha=3.0)(self.semantic_encoder_layer[i](edata)) * 
                                nn.Sigmoid()(self.semantic_encoder_layer[i](edata)))
            
            # Aggregate messages for each node type
            semantic_embeddings['g'].append(
                self.message_aggregator_layer[0](new_g.nodes('g').tolist(), metapath_instances, 
                                                metapathembedding, h['g']))
            semantic_embeddings['m'].append(
                self.message_aggregator_layer[1](new_g.nodes('m').tolist(), metapath_instances, 
                                                metapathembedding, h['m']))
            semantic_embeddings['d'].append(
                self.message_aggregator_layer[2](new_g.nodes('d').tolist(), metapath_instances, 
                                                metapathembedding, h['d']))
        
        # Fuse embeddings from different meta-paths for each node type
        for ntype in semantic_embeddings.keys():
            if semantic_embeddings[ntype]:
                semantic_embeddings[ntype] = self.stack_embedding(semantic_embeddings[ntype])
                if ntype == 'g':
                    nodes_embeddings[ntype], g_beta = self.subgraph_fusion(semantic_embeddings[ntype])
                elif ntype == 'm':
                    nodes_embeddings[ntype], m_beta = self.subgraph_fusion(semantic_embeddings[ntype])
                elif ntype == 'd':
                    nodes_embeddings[ntype], d_beta = self.subgraph_fusion(semantic_embeddings[ntype])
        
        return nodes_embeddings

class InformationBottleneck(nn.Module):
    """Information Bottleneck layer for representation learning"""
    def __init__(self, dropout):
        super(InformationBottleneck, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(9216)
        self.fc1 = nn.Linear(9216, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.vec_mean = nn.Linear(2048, 2048)
        self.vec_cov = nn.Linear(2048, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        
    def forward(self, pair):
        """Apply information bottleneck"""
        pair = self.dropout(self.bn1(pair))
        pair = self.bn2(nn.CELU(alpha=3.0)(self.fc1(pair)))
        pair = self.bn2(nn.CELU(alpha=3.0)(self.fc2(pair)))
        pair = self.dropout(pair)
        
        # Calculate mean and covariance
        vec_mean, vec_cov = self.bn3(self.vec_mean(pair)), F.softplus(self.bn3(self.vec_cov(pair)) - 5)
        
        # Reparameterization sampling
        eps = torch.randn_like(vec_cov)
        jointF = vec_mean + vec_cov * eps
        
        return jointF, vec_mean, vec_cov

class MLPDecoder(nn.Module):
    """MLP decoder for final prediction"""
    def __init__(self, dropout, hidden_size, num_heads, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)
        self.drop = dropout

    def forward(self, x):
        """Decode to prediction"""
        x = self.bn1(nn.CELU(alpha=3.0)(self.fc1(x)))
        x = nn.Dropout(self.drop)(x)
        x = self.bn2(nn.CELU(alpha=3.0)(self.fc2(x)))
        x = self.fc4(x)
        return x