import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv, BatchNorm
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from py2neo import Graph
import pickle
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ESGGraphDataset(Dataset):
    def __init__(self, graph, transform=None, pre_transform=None, embedding_model='all-MiniLM-L6-v2'):
        self.graph = graph
        self.embedding_model = SentenceTransformer(embedding_model)
        super(ESGGraphDataset, self).__init__(root='data/esg_dataset', transform=transform, pre_transform=pre_transform)
        self.data = None
        self.process()
        
    @property
    def raw_file_names(self):
        return []
        
    @property
    def processed_file_names(self):
        return ['esg_data.pt']
    
    def download(self):
        pass
    
    # Extract features from sentence nodes using a pre-trained sentence transformer
    def _build_node_features(self, sentences):
        print("Building node features using sentence embeddings...")
        # getting all sentence texts
        texts = [s['text'] for s in sentences]
        # generating embeddings for all sentences in batches
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        # converting to tensor
        X = torch.FloatTensor(embeddings)
        esg_features = torch.zeros((len(sentences), 3))  # env, social and gov
        
        for i, s in enumerate(sentences):
            text = s['text'].lower()
            # Environmental signals
            env_keywords = ['environmental', 'climate', 'carbon', 'emission', 'renewable', 
                           'sustainable', 'energy', 'water', 'waste', 'recycling', 'biodiversity']
            # Social signals
            social_keywords = ['social', 'employee', 'diversity', 'inclusion', 'community', 
                              'human rights', 'labor', 'health', 'safety', 'customer', 'privacy']
            # Governance signals
            gov_keywords = ['governance', 'board', 'compliance', 'ethics', 'transparency', 
                           'risk', 'audit', 'executive', 'compensation', 'shareholder', 'accountability']
            
            # count keyword occurrences
            env_count = sum(1 for word in env_keywords if word in text)
            social_count = sum(1 for word in social_keywords if word in text)
            gov_count = sum(1 for word in gov_keywords if word in text)
            
            # normalizing counts
            total = max(1, env_count + social_count + gov_count)
            esg_features[i, 0] = env_count / total
            esg_features[i, 1] = social_count / total  
            esg_features[i, 2] = gov_count / total
        
        # concatenate sentence embeddings with ESG domain features
        X = torch.cat([X, esg_features], dim=1)
        print(f"Node features shape: {X.shape}")
        return X

    
    def _build_edge_index(self, sentence_nodes, embeddings=None, similarity_threshold=0.6, max_connections=10):
        print("Building edge index based on semantic similarity...")
        edge_index = []
        edge_weights = []
        # Get sentence texts and convert embeddings to numpy for similarity calculation
        if embeddings is None:
            texts = [node['text'] for node in sentence_nodes]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        print("Computing semantic similarities...")
        num_nodes = len(sentence_nodes)
        batch_size = 100  # Process in batches to avoid memory issues
        for i in range(0, num_nodes, batch_size):
            end_i = min(i + batch_size, num_nodes)
            batch_embeddings = embeddings[i:end_i]
            similarities = cosine_similarity(batch_embeddings, embeddings)
            # For each node in the batch, add edges to most similar nodes
            for batch_idx, global_idx in enumerate(range(i, end_i)):
                sim_scores = similarities[batch_idx]
                sim_scores[global_idx] = 0  # removing self-similarity
                top_indices = np.argsort(sim_scores)[-max_connections:]
                top_scores = sim_scores[top_indices]
                for j, score in zip(top_indices, top_scores):
                    if score >= similarity_threshold:
                        edge_index.append([global_idx, j])
                        edge_weights.append(float(score))
        # converting to tensor
        edge_index = torch.LongTensor(edge_index).t()
        edge_weights = torch.FloatTensor(edge_weights)
        print(f"Created edge index with {edge_index.shape[1]} edges")
        return edge_index, edge_weights
 
    # Get ESG scores for each sentence with data augmentation
    def _get_sentence_labels(self, sentence_nodes):
        print("Getting sentence labels...")
        y = np.zeros((len(sentence_nodes), 3))
        # get ESG scores for each sentence
        for i, node in enumerate(sentence_nodes):
            # query for ESG scores
            scores = self.graph.run("""MATCH (s:Sentence {id: $id})-[r:CATEGORIZED_AS]->(c:Category) RETURN c.name as category, r.score as score""", id=node['id']).data()
            # assign scores to appropriate position in label matrix
            for score_data in scores:
                if score_data['category'] == 'Environmental':
                    y[i, 0] = score_data['score']
                elif score_data['category'] == 'Social':
                    y[i, 1] = score_data['score']
                elif score_data['category'] == 'Governance':
                    y[i, 2] = score_data['score']
        return torch.FloatTensor(y)
    
    # process the Neo4j data into a PyTorch Geometric graph
    def process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        # Load data from Neo4j - all sentences
        sentence_nodes = list(self.graph.run("""MATCH (s:Sentence) RETURN s.id as id, s.text as text""").data())
        print(f"Loaded {len(sentence_nodes)} sentence nodes from Neo4j")
        if not sentence_nodes:
            raise ValueError("No sentence nodes found in the database!")
        # build features
        x = self._build_node_features(sentence_nodes)
        # Bbuild edge index with weights based on semantic similarity
        edge_index, edge_weights = self._build_edge_index(sentence_nodes, embeddings=x[:, :-3])
        # get ESG scores as labels
        y = self._get_sentence_labels(sentence_nodes)
        # create PyG data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights, y=y)
        self.data = data
        torch.save(data, os.path.join(self.processed_dir, 'esg_data.pt'))
        # save node mapping for later reference
        with open(os.path.join(self.processed_dir, 'node_mapping.pkl'), 'wb') as f:
            pickle.dump({i: node['id'] for i, node in enumerate(sentence_nodes)}, f)
        print("Data processing complete!")
    
    def get(self, idx):
        return self.data

# Graph Neural Network for ESG score prediction
class ImprovedESGGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3, dropout=0.3, 
                 architecture='multi', num_layers=3, heads=4, use_edge_weights=True):
        super(ImprovedESGGNN, self).__init__()
        self.architecture = architecture
        self.use_edge_weights = use_edge_weights
        # input projection
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        
        # different architecture options
        if architecture == 'gcn':
            # GCN layers
            self.conv_layers = torch.nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
            ])
        elif architecture == 'sage':
            # GraphSAGE layers
            self.conv_layers = torch.nn.ModuleList([
                SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)
            ])
        elif architecture == 'gat':
            # GAT layers
            self.conv_layers = torch.nn.ModuleList([
                GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True) 
                for _ in range(num_layers)
            ])
        elif architecture == 'transformer':
            # Graph Transformer layers
            self.conv_layers = torch.nn.ModuleList([
                TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)
                for _ in range(num_layers)
            ])
        elif architecture == 'multi':
            # multi-architecture
            self.conv_layers = torch.nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim),
                SAGEConv(hidden_dim, hidden_dim),
                GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)
            ])
        # batch normalization layers
        self.batch_norms = torch.nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers)])
        self.has_skip = True
        # output MLP
        self.out_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim)
        )
        # dropout
        self.dropout = dropout

    # forward pass
    def forward(self, x, edge_index, edge_attr=None):
        # input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # initial representation
        x_init = x
        # apply convolutional layers
        if self.architecture == 'multi':
            # For multi-architecture, apply different convs sequentially
            for i, conv in enumerate(self.conv_layers):
                if self.use_edge_weights and edge_attr is not None and isinstance(conv, GCNConv):
                    x_conv = conv(x, edge_index, edge_weight=edge_attr)
                else:
                    x_conv = conv(x, edge_index)     
                x_conv = self.batch_norms[i](x_conv)
                x_conv = F.relu(x_conv)
                x_conv = F.dropout(x_conv, p=self.dropout, training=self.training)
                # skip connection
                if self.has_skip and x_conv.shape == x.shape:
                    x = x_conv + x
                else:
                    x = x_conv
        else:
            # for single architecture, apply layers with residual connections
            for i, conv in enumerate(self.conv_layers):
                if self.use_edge_weights and edge_attr is not None and isinstance(conv, GCNConv):
                    x_conv = conv(x, edge_index, edge_weight=edge_attr)
                else:
                    x_conv = conv(x, edge_index)
                x_conv = self.batch_norms[i](x_conv)
                x_conv = F.relu(x_conv)
                x_conv = F.dropout(x_conv, p=self.dropout, training=self.training)
                
                # skip connection
                if self.has_skip and x_conv.shape == x.shape:
                    x = x_conv + x
                else:
                    x = x_conv
        if x.shape == x_init.shape:
            x = x + x_init
        x = self.out_mlp(x)
        return x

# class for training and evaluating the ESG GNN model
class ImprovedESGGraphTrainer:
    # initialize the trainer
    def __init__(self, model, device='cuda', lr=0.001, weight_decay=1e-4, scheduler_factor=0.5, scheduler_patience=10):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=scheduler_factor, patience=scheduler_patience, verbose=True
        )
    # train for one epoch
    def train_epoch(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_attr = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = data.edge_attr.to(self.device)
        # forward pass
        out = self.model(x, edge_index, edge_attr)
        # calculate loss only on training nodes
        if hasattr(data, 'train_mask'):
            # apply same mask to both predictions and targets
            loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask].to(self.device))
        else:
            loss = F.mse_loss(out, data.y.to(self.device))
        # backward pass
        loss.backward()
        # gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()
    
    # evaluate the model
    def evaluate(self, data):
        self.model.eval()    
        with torch.no_grad():
            # Move data to device
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            edge_attr = None
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                edge_attr = data.edge_attr.to(self.device)
            
            out = self.model(x, edge_index, edge_attr)
            loss = F.mse_loss(out, data.y.to(self.device))
            # move tensors to CPU for evaluation metrics
            pred = out.detach().cpu().numpy()
            true = data.y.detach().cpu().numpy()
            # calculate RMSE and R² for each ESG dimension
            metrics = {
                'loss': loss.item(),
                'env_rmse': np.sqrt(mean_squared_error(true[:, 0], pred[:, 0])),
                'social_rmse': np.sqrt(mean_squared_error(true[:, 1], pred[:, 1])),
                'gov_rmse': np.sqrt(mean_squared_error(true[:, 2], pred[:, 2])),
                'env_r2': r2_score(true[:, 0], pred[:, 0]),
                'social_r2': r2_score(true[:, 1], pred[:, 1]),
                'gov_r2': r2_score(true[:, 2], pred[:, 2]),
                'overall_rmse': np.sqrt(mean_squared_error(true.flatten(), pred.flatten())),
                'overall_r2': r2_score(true.flatten(), pred.flatten())
            }
            return metrics, pred
    
    # cross-validation training
    def cross_validation(self, data, k_folds=5, epochs=200, patience=30):
        print(f"Performing {k_folds}-fold cross-validation")
        num_nodes = data.x.size(0)
        cv_results = {
            'env_rmse': [],
            'social_rmse': [],
            'gov_rmse': [],
            'env_r2': [],
            'social_r2': [],
            'gov_r2': [],
            'overall_rmse': [],
            'overall_r2': []
        }
        # setting up k-fold cross validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_indices = list(kf.split(range(num_nodes)))
        # for each fold
        for fold, (train_idx, val_idx) in enumerate(fold_indices):
            print(f"\nFold {fold+1}/{k_folds}")
            # reset model parameters
            for layer in self.model.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            # optimizers and schedulers
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.5, patience=10, verbose=True
            )
            # create masks
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            data.train_mask = train_mask
            data.val_mask = val_mask
            # for early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            best_metrics = None
            # training loop
            for epoch in range(epochs):
                train_loss = self.train_epoch(data)
                val_metrics, _ = self.evaluate(data.clone())
                val_loss = val_metrics['loss']
                self.scheduler.step(val_loss)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                    print(f"E: RMSE={val_metrics['env_rmse']:.4f}, R²={val_metrics['env_r2']:.4f} | "
                          f"S: RMSE={val_metrics['social_rmse']:.4f}, R²={val_metrics['social_r2']:.4f} | "
                          f"G: RMSE={val_metrics['gov_rmse']:.4f}, R²={val_metrics['gov_r2']:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {key: value.cpu() for key, value in self.model.state_dict().items()}
                    best_metrics = val_metrics
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            # add best metrics from this fold to overall results
            for key in cv_results.keys():
                cv_results[key].append(best_metrics[key])
            # restore best model for this fold
            self.model.load_state_dict(best_model_state)
        # calculate average metrics across all folds
        avg_results = {key: np.mean(values) for key, values in cv_results.items()}
        std_results = {key: np.std(values) for key, values in cv_results.items()}
        print("\nCross-validation results:")
        print(f"Environmental: RMSE={avg_results['env_rmse']:.4f}±{std_results['env_rmse']:.4f}, "
              f"R²={avg_results['env_r2']:.4f}±{std_results['env_r2']:.4f}")
        print(f"Social: RMSE={avg_results['social_rmse']:.4f}±{std_results['social_rmse']:.4f}, "
              f"R²={avg_results['social_r2']:.4f}±{std_results['social_r2']:.4f}")
        print(f"Governance: RMSE={avg_results['gov_rmse']:.4f}±{std_results['gov_rmse']:.4f}, "
              f"R²={avg_results['gov_r2']:.4f}±{std_results['gov_r2']:.4f}")
        print(f"Overall: RMSE={avg_results['overall_rmse']:.4f}±{std_results['overall_rmse']:.4f}, "
              f"R²={avg_results['overall_r2']:.4f}±{std_results['overall_r2']:.4f}")
        return avg_results, std_results
    
    # train the model
    def train(self, data, epochs=300, validation_split=0.2, patience=30):
        print(f"Training on {self.device}")
        # create train/val masks
        num_nodes = data.x.size(0)
        indices = list(range(num_nodes))
        train_idx, val_idx = train_test_split(indices, test_size=validation_split, random_state=42)
        # create train and validation masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        data.train_mask = train_mask
        data.val_mask = val_mask
        history = {
            'train_loss': [],
            'val_loss': [],
            'env_rmse': [],
            'social_rmse': [],
            'gov_rmse': [],
            'env_r2': [],
            'social_r2': [],
            'gov_r2': [],
            'overall_rmse': [],
            'overall_r2': [],
            'learning_rates': []
        }
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        best_epoch = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(data)
            val_metrics, _ = self.evaluate(data)
            val_loss = val_metrics['loss']
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['env_rmse'].append(val_metrics['env_rmse'])
            history['social_rmse'].append(val_metrics['social_rmse'])
            history['gov_rmse'].append(val_metrics['gov_rmse'])
            history['env_r2'].append(val_metrics['env_r2'])
            history['social_r2'].append(val_metrics['social_r2'])
            history['gov_r2'].append(val_metrics['gov_r2'])
            history['overall_rmse'].append(val_metrics['overall_rmse'])
            history['overall_r2'].append(val_metrics['overall_r2'])
            history['learning_rates'].append(current_lr)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
                print(f"E: RMSE={val_metrics['env_rmse']:.4f}, R²={val_metrics['env_r2']:.4f} | "
                      f"S: RMSE={val_metrics['social_rmse']:.4f}, R²={val_metrics['social_r2']:.4f} | "
                      f"G: RMSE={val_metrics['gov_rmse']:.4f}, R²={val_metrics['gov_r2']:.4f}")
                print(f"Overall: RMSE={val_metrics['overall_rmse']:.4f}, R²={val_metrics['overall_r2']:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch
                best_model_state = {key: value.cpu() for key, value in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}, best epoch was {best_epoch+1}")
                    break
            # stop if learning rate becomes too small
            if current_lr < 1e-6:
                print(f"Learning rate too small ({current_lr:.8f}), stopping at epoch {epoch+1}")
                break
        # restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        return history
    
    def save_model(self, path, include_optimizer=True):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_architecture': self.model.architecture if hasattr(self.model, 'architecture') else 'unknown'
        }
        if include_optimizer:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
            save_dict['scheduler_state_dict'] = self.scheduler.state_dict() if self.scheduler else None   
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
        
    # load the model
    def load_model(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])   
        print(f"Model loaded from {path}")
        return checkpoint.get('model_architecture', 'unknown')
        
    def plot_training_history(self, history):
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        axs[0, 0].plot(history['train_loss'], label='Train Loss')
        axs[0, 0].plot(history['val_loss'], label='Val Loss')
        axs[0, 0].set_title('Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('MSE Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        # RMSE
        axs[0, 1].plot(history['env_rmse'], label='Environmental')
        axs[0, 1].plot(history['social_rmse'], label='Social')
        axs[0, 1].plot(history['gov_rmse'], label='Governance')
        axs[0, 1].plot(history['overall_rmse'], label='Overall', linestyle='--', color='black')
        axs[0, 1].set_title('RMSE')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('RMSE')
        axs[0, 1].legend()
        axs[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # R²
        axs[0, 2].plot(history['env_r2'], label='Environmental')
        axs[0, 2].plot(history['social_r2'], label='Social')
        axs[0, 2].plot(history['gov_r2'], label='Governance')
        axs[0, 2].plot(history['overall_r2'], label='Overall', linestyle='--', color='black')
        axs[0, 2].set_title('R²')
        axs[0, 2].set_xlabel('Epoch')
        axs[0, 2].set_ylabel('R²')
        axs[0, 2].legend()
        axs[0, 2].grid(True, linestyle='--', alpha=0.7)
        
        if 'learning_rates' in history:
            axs[1, 0].plot(history['learning_rates'], marker='o', markersize=3)
            axs[1, 0].set_title('Learning Rate')
            axs[1, 0].set_xlabel('Epoch')
            axs[1, 0].set_ylabel('Learning Rate')
            axs[1, 0].set_yscale('log')
            axs[1, 0].grid(True, linestyle='--', alpha=0.7)
        env_social = [history['env_rmse'][i] / max(0.0001, history['social_rmse'][i]) 
                     for i in range(len(history['env_rmse']))]
        env_gov = [history['env_rmse'][i] / max(0.0001, history['gov_rmse'][i]) 
                  for i in range(len(history['env_rmse']))]
        social_gov = [history['social_rmse'][i] / max(0.0001, history['gov_rmse'][i]) 
                     for i in range(len(history['social_rmse']))]
        axs[1, 1].plot(env_social, label='Env/Social')
        axs[1, 1].plot(env_gov, label='Env/Gov')
        axs[1, 1].plot(social_gov, label='Social/Gov')
        axs[1, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        axs[1, 1].set_title('ESG Balance (RMSE Ratios)')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Ratio')
        axs[1, 1].legend()
        axs[1, 1].grid(True, linestyle='--', alpha=0.7)
        axs[1, 2].axis('off')
        axs[1, 2].text(0.5, 0.9, 'Best Metrics:', horizontalalignment='center', fontsize=12, fontweight='bold')
        best_epoch = np.argmin(history['val_loss'])
        metrics_text = (
            f"Best Epoch: {best_epoch}\n\n"
            f"Environmental:\n"
            f"  RMSE: {history['env_rmse'][best_epoch]:.4f}\n"
            f"  R²: {history['env_r2'][best_epoch]:.4f}\n\n"
            f"Social:\n"
            f"  RMSE: {history['social_rmse'][best_epoch]:.4f}\n"
            f"  R²: {history['social_r2'][best_epoch]:.4f}\n\n"
            f"Governance:\n"
            f"  RMSE: {history['gov_rmse'][best_epoch]:.4f}\n"
            f"  R²: {history['gov_r2'][best_epoch]:.4f}\n\n"
            f"Overall:\n"
            f"  RMSE: {history['overall_rmse'][best_epoch]:.4f}\n"
            f"  R²: {history['overall_r2'][best_epoch]:.4f}"
        )
        axs[1, 2].text(0.5, 0.5, metrics_text, horizontalalignment='center', verticalalignment='center', fontsize=10)
        plt.tight_layout()
        plt.savefig('runs/training_history_improved.png', dpi=300)
        plt.show()
        
    # Predict ESG scores for a document
    def predict_document_scores(self, graph, document_id, model_path=None):
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        self.model.eval()
        # get sentences from document
        sentences = graph.run("""MATCH (d:Document {filename: $doc_id})-[:CONTAINS]->(s:Sentence) RETURN s.id as id, s.text as text""", doc_id=document_id).data()
        if not sentences:
            print(f"No sentences found for document {document_id}")
            return None
        print(f"Predicting ESG scores for {len(sentences)} sentences in document {document_id}")
        # create sentence transformer for embeddings
        if not hasattr(self, 'embedding_model'):
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # process sentences
        texts = [s['text'] for s in sentences]
        # generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        # add ESG domain knowledge as additional features
        num_sentences = len(sentences)
        esg_features = np.zeros((num_sentences, 3))
        for i, s in enumerate(sentences):
            text = s['text'].lower()
            env_keywords = ['environmental', 'climate', 'carbon', 'emission', 'renewable', 'sustainable', 'energy', 'water', 'waste', 'recycling', 'biodiversity']
            social_keywords = ['social', 'employee', 'diversity', 'inclusion', 'community', 'human rights', 'labor', 'health', 'safety', 'customer', 'privacy']
            gov_keywords = ['governance', 'board', 'compliance', 'ethics', 'transparency', 'risk', 'audit', 'executive', 'compensation', 'shareholder', 'accountability']
            env_count = sum(1 for word in env_keywords if word in text)
            social_count = sum(1 for word in social_keywords if word in text)
            gov_count = sum(1 for word in gov_keywords if word in text)
            total = max(1, env_count + social_count + gov_count)
            esg_features[i, 0] = env_count / total
            esg_features[i, 1] = social_count / total  
            esg_features[i, 2] = gov_count / total
        X = np.hstack((embeddings, esg_features))
        edge_index = []
        edge_weights = []
        similarities = cosine_similarity(embeddings)
        similarity_threshold = 0.6
        max_connections = 10
        for i in range(num_sentences):
            sim_scores = similarities[i].copy()
            sim_scores[i] = 0
            top_indices = np.argsort(sim_scores)[-max_connections:]
            top_scores = sim_scores[top_indices]
            for j, score in zip(top_indices, top_scores):
                if score >= similarity_threshold:
                    edge_index.append([i, j])
                    edge_weights.append(float(score))
        if not edge_index:
            print("Warning: No edges created based on similarity threshold. Creating minimal connections.")
            for i in range(num_sentences):
                if i > 0:
                    edge_index.append([i, i-1])
                    edge_weights.append(0.5)
                if i < num_sentences - 1:
                    edge_index.append([i, i+1])
                    edge_weights.append(0.5)
        x = torch.FloatTensor(X)
        edge_index = torch.LongTensor(edge_index).t()
        edge_weights = torch.FloatTensor(edge_weights)
        with torch.no_grad():
            self.model.eval()
            out = self.model(x.to(self.device), edge_index.to(self.device), edge_weights.to(self.device))
            sentence_scores = out.cpu().numpy()
            weights = np.abs(sentence_scores - 0.5) + 0.5
            weighted_scores = sentence_scores * weights
            avg_scores = weighted_scores.sum(axis=0) / weights.sum(axis=0)
            simple_avg_scores = sentence_scores.mean(axis=0)
            return {
                'weighted_scores': {
                    'environmental_score': float(avg_scores[0]),
                    'social_score': float(avg_scores[1]),
                    'governance_score': float(avg_scores[2])
                },
                'average_scores': {
                    'environmental_score': float(simple_avg_scores[0]),
                    'social_score': float(simple_avg_scores[1]),
                    'governance_score': float(simple_avg_scores[2])
                },
                'sentence_level_predictions': [
                    {
                        'text': sentences[i]['text'][:100] + '...' if len(sentences[i]['text']) > 100 else sentences[i]['text'],
                        'environmental': float(sentence_scores[i][0]),
                        'social': float(sentence_scores[i][1]),
                        'governance': float(sentence_scores[i][2])
                    }
                    for i in range(len(sentences))
                ]
            }


def main():
    # Connect to Neo4j
    NEO4J_URI = "bolt://localhost:7687"  
    NEO4J_USER = "neo4j"              
    NEO4J_PASSWORD = "12345678"          
    
    try:
        graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        print("Connected to Neo4j database")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return
    
    # Try loading or creating dataset
    try:
        # Create dataset with real sentence embeddings
        print("Creating dataset with sentence transformer embeddings...")
        dataset = ESGGraphDataset(graph)
        data = dataset.get(0)
        
        print(f"Dataset created with {data.x.shape[0]} nodes and {data.edge_index.shape[1]} edges")
        print(f"Feature dimension: {data.x.shape[1]}")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return
    
    # Configure model architecture
    architectures = ['multi', 'gcn', 'sage', 'gat', 'transformer']
    selected_architecture = architectures[0]  # Change index to try different architectures
    
    # Define hyperparameters
    hidden_dim = 128
    num_layers = 3
    heads = 4
    dropout = 0.3
    learning_rate = 0.001
    weight_decay = 1e-4
    
    print(f"Creating {selected_architecture} model with {hidden_dim} hidden dimensions, "
          f"{num_layers} layers, and dropout {dropout}")
    
    # Create and train model
    model = ImprovedESGGNN(
        input_dim=data.x.shape[1],
        hidden_dim=hidden_dim,
        output_dim=data.y.shape[1],
        dropout=dropout,
        architecture=selected_architecture,
        num_layers=num_layers,
        heads=heads,
        use_edge_weights=True
    )
    # Create trainer
    trainer = ImprovedESGGraphTrainer(
        model, 
        device='cuda', 
        lr=learning_rate, 
        weight_decay=weight_decay,
        scheduler_factor=0.5,
        scheduler_patience=10
    )
    # Training options
    run_cv = False  # Set to True to run cross-validation
    if run_cv:
        # Run cross-validation
        print("\nRunning cross-validation...")
        avg_results, std_results = trainer.cross_validation(
            data, 
            k_folds=5, 
            epochs=200, 
            patience=20
        )
    # Train the model
    print("\nTraining final model...")
    history = trainer.train(
        data, 
        epochs=300, 
        validation_split=0.2, 
        patience=30
    )
    # Save the model
    model_path = f'models/esg_gnn_{selected_architecture}_model.pt'
    trainer.save_model(model_path)
    # Plot training history
    trainer.plot_training_history(history)
    # Predict scores for available documents
    try:
        # Get document filenames from database
        doc_results = graph.run("MATCH (d:Document) RETURN d.filename as filename").data()
        
        if doc_results:
            for doc in doc_results:
                document_id = doc['filename']
                print(f"\nPredicting ESG scores for document: {document_id}")
                
                scores = trainer.predict_document_scores(graph, document_id)
                
                if scores:
                    print("\nWeighted ESG Scores:")
                    w_scores = scores['weighted_scores']
                    print(f"Environmental: {w_scores['environmental_score']:.4f}")
                    print(f"Social: {w_scores['social_score']:.4f}")
                    print(f"Governance: {w_scores['governance_score']:.4f}")
                    
                    print("\nAverage ESG Scores:")
                    a_scores = scores['average_scores']
                    print(f"Environmental: {a_scores['environmental_score']:.4f}")
                    print(f"Social: {a_scores['social_score']:.4f}")
                    print(f"Governance: {a_scores['governance_score']:.4f}")
                    
                    # Print top 3 sentences for each ESG category
                    print("\nTop Environmental Sentences:")
                    env_sorted = sorted(scores['sentence_level_predictions'], 
                                      key=lambda x: x['environmental'], reverse=True)[:3]
                    for i, s in enumerate(env_sorted):
                        print(f"{i+1}. {s['text']} (Score: {s['environmental']:.4f})")
                    
                    print("\nTop Social Sentences:")
                    social_sorted = sorted(scores['sentence_level_predictions'], 
                                         key=lambda x: x['social'], reverse=True)[:3]
                    for i, s in enumerate(social_sorted):
                        print(f"{i+1}. {s['text']} (Score: {s['social']:.4f})")
                    
                    print("\nTop Governance Sentences:")
                    gov_sorted = sorted(scores['sentence_level_predictions'], 
                                      key=lambda x: x['governance'], reverse=True)[:3]
                    for i, s in enumerate(gov_sorted):
                        print(f"{i+1}. {s['text']} (Score: {s['governance']:.4f})")
        else:
            print("No documents found in database")
    except Exception as e:
        print(f"Error predicting document scores: {e}")

    print("\nTraining and evaluation complete!")
    
    # Provide additional insights
    print("\nModel analysis and recommendations:")
    if 'env_r2' in history and history['env_r2']:
        best_epoch = np.argmax(history['overall_r2'])
        best_r2 = history['overall_r2'][best_epoch]
        if best_r2 < 0:
            print("- WARNING: The model is still producing negative R² values, indicating fundamental issues.")
            print("  Consider collecting more labeled data or improving feature extraction.")
        elif best_r2 < 0.3:
            print("- The model shows some predictive ability but could be substantially improved.")
            print("  Try increasing model complexity or improving the graph structure.")
        else:
            print(f"- Good model performance achieved with R² of {best_r2:.4f}!")
            
    print("- For production use, consider fine-tuning on domain-specific data.")
    print("- Experiment with different text embedding models if performance is still inadequate.")

