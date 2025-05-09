import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from py2neo import Graph
import pickle
import random

# Dataset for loading ESG graph data from Neo4j
class ESGGraphDataset(Dataset):
    
    #Initialize the dataset
    def __init__(self, graph, transform=None, pre_transform=None):
        self.graph = graph
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
    
    #Extract features from sentence nodes
    def _build_node_features(self, sentences):
        print("Building node features...")
        
        # Initialize feature matrix
        num_sentences = len(sentences)
        # Use sentence text embeddings as features
        feature_dim = 64  # Size of embedding
        X = np.zeros((num_sentences, feature_dim))
        
        # Build features from sentences
        for i, s in enumerate(tqdm(sentences)):
            # Generate deterministic seed from sentence text for reproducibility
            seed = hash(s['text']) % 10000
            np.random.seed(seed)
            
            embedding = np.random.rand(feature_dim)
            
            if 'environmental' in s['text'].lower():
                embedding[:20] += 0.3
            if 'social' in s['text'].lower():
                embedding[20:40] += 0.3
            if 'governance' in s['text'].lower():
                embedding[40:] += 0.3
                
            X[i] = embedding
        
        return torch.FloatTensor(X)
    
    #Build edge index from sentence relationships
    def _build_edge_index(self, sentence_nodes):
        print("Building edge index...")
        
        # Create node ID mapping
        node_id_map = {node['id']: i for i, node in enumerate(sentence_nodes)}
        
        # Get relationships between sentences
        edges = self.graph.run("""
            MATCH (s1:Sentence)<-[:CONTAINS]-(d:Document)-[:CONTAINS]->(s2:Sentence)
            WHERE s1 <> s2
            RETURN s1.id as source, s2.id as target
        """).data()
        
        if not edges:
            print("Warning: No edges found! Creating a fully connected graph as fallback.")
            # Create a fully connected graph as fallback
            ids = [node['id'] for node in sentence_nodes]
            edges = []
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    edges.append({'source': ids[i], 'target': ids[j]})
        
        # Create edge index
        edge_index = []
        for edge in edges:
            source = node_id_map.get(edge['source'])
            target = node_id_map.get(edge['target'])
            if source is not None and target is not None:
                edge_index.append([source, target])
                edge_index.append([target, source])  # Make it undirected
        
        # Convert to tensor
        edge_index = torch.LongTensor(edge_index).t()
        
        print(f"Created edge index with {edge_index.shape[1]} edges")
        return edge_index
    
    # Get ESG scores for each sentence
    def _get_sentence_labels(self, sentence_nodes):
        print("Getting sentence labels...")
        
        y = np.zeros((len(sentence_nodes), 3))  # [env_score, social_score, gov_score]
        
        # For each sentence, get its ESG scores
        for i, node in enumerate(sentence_nodes):
            # Query for ESG scores
            scores = self.graph.run("""
                MATCH (s:Sentence {id: $id})-[r:CATEGORIZED_AS]->(c:Category)
                RETURN c.name as category, r.score as score
            """, id=node['id']).data()
            
            # Assign scores to appropriate position in label matrix
            for score_data in scores:
                if score_data['category'] == 'Environmental':
                    y[i, 0] = score_data['score']
                elif score_data['category'] == 'Social':
                    y[i, 1] = score_data['score']
                elif score_data['category'] == 'Governance':
                    y[i, 2] = score_data['score']
        
        return torch.FloatTensor(y)
    
    #Process the Neo4j data into a PyTorch Geometric graph
    def process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
        # Load data from Neo4j - all sentences
        sentence_nodes = list(self.graph.run("""
            MATCH (s:Sentence)
            RETURN s.id as id, s.text as text
        """).data())
        
        print(f"Loaded {len(sentence_nodes)} sentence nodes from Neo4j")
        
        if not sentence_nodes:
            raise ValueError("No sentence nodes found in the database!")
            
        # Build features
        x = self._build_node_features(sentence_nodes)
        
        # Build edge index
        edge_index = self._build_edge_index(sentence_nodes)
        
        # Get ESG scores as labels
        y = self._get_sentence_labels(sentence_nodes)
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        self.data = data
        
        # Save data
        torch.save(data, os.path.join(self.processed_dir, 'esg_data.pt'))
        
        # Save node mapping for later reference
        with open(os.path.join(self.processed_dir, 'node_mapping.pkl'), 'wb') as f:
            pickle.dump({i: node['id'] for i, node in enumerate(sentence_nodes)}, f)
        
        print("Data processing complete!")
    
    def len(self):
        return 1
    
    def get(self, idx):
        return self.data

# Graph Neural Network for ESG score prediction
class ESGGNN(torch.nn.Module):
    
    # Initialize the GNN
    def __init__(self, input_dim, hidden_dim=64, output_dim=3, dropout=0.2):
        super(ESGGNN, self).__init__()
        
        # Graph convolutional layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Output layer
        self.out = torch.nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout = dropout
        
    # Forward pass
    def forward(self, x, edge_index):
        # First graph conv layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second graph conv layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Output layer
        x = self.out(x)
        
        return x

#Class for training and evaluating the ESG GNN model
class ESGGraphTrainer:
    
    #Initialize the trainer
    def __init__(self, model, device='cuda', lr=0.001, weight_decay=5e-4):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
      
    #Train for one epoch
    def train_epoch(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass for all nodes
        out = self.model(data.x.to(self.device), data.edge_index.to(self.device))
        
        # Calculate loss only on training nodes
        if hasattr(data, 'train_mask'):
            # Apply same mask to both predictions and targets
            loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask].to(self.device))
        else:
            loss = F.mse_loss(out, data.y.to(self.device))
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    #Evaluate the model
    def evaluate(self, data):
        self.model.eval()    
        with torch.no_grad():
            out = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            loss = F.mse_loss(out, data.y.to(self.device))
            
            # Move tensors to CPU for evaluation metrics
            pred = out.detach().cpu().numpy()
            true = data.y.detach().cpu().numpy()
            
            # Calculate RMSE and R² for each ESG dimension
            metrics = {
                'loss': loss.item(),
                'env_rmse': np.sqrt(mean_squared_error(true[:, 0], pred[:, 0])),
                'social_rmse': np.sqrt(mean_squared_error(true[:, 1], pred[:, 1])),
                'gov_rmse': np.sqrt(mean_squared_error(true[:, 2], pred[:, 2])),
                'env_r2': r2_score(true[:, 0], pred[:, 0]),
                'social_r2': r2_score(true[:, 1], pred[:, 1]),
                'gov_r2': r2_score(true[:, 2], pred[:, 2])
            }
            
            return metrics, pred
    
    #Train the model
    def train(self, data, epochs=200, validation_split=0.2, patience=20):
        print(f"Training on {self.device}")
        
        # Create train/val masks
        num_nodes = data.x.size(0)
        indices = list(range(num_nodes))
        train_idx, val_idx = train_test_split(indices, test_size=validation_split, random_state=42)
        
        # Create train and validation masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'env_rmse': [],
            'social_rmse': [],
            'gov_rmse': [],
            'env_r2': [],
            'social_r2': [],
            'gov_r2': []
        }
        
        # For early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Move data to device
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = data.y.to(self.device)
        
        # Training loop
        for epoch in range(epochs):
            # Train phase
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass for all nodes
            out = self.model(x, edge_index)
            
            # Calculate loss only on training nodes
            train_loss = F.mse_loss(out[train_mask], y[train_mask])
            
            # Backward pass
            train_loss.backward()
            self.optimizer.step()
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                out = self.model(x, edge_index)
                val_loss = F.mse_loss(out[val_mask], y[val_mask])
                
                # Calculate metrics
                pred_val = out[val_mask].cpu().numpy()
                true_val = y[val_mask].cpu().numpy()
                
                env_rmse = np.sqrt(mean_squared_error(true_val[:, 0], pred_val[:, 0]))
                social_rmse = np.sqrt(mean_squared_error(true_val[:, 1], pred_val[:, 1]))
                gov_rmse = np.sqrt(mean_squared_error(true_val[:, 2], pred_val[:, 2]))
                
                env_r2 = r2_score(true_val[:, 0], pred_val[:, 0])
                social_r2 = r2_score(true_val[:, 1], pred_val[:, 1])
                gov_r2 = r2_score(true_val[:, 2], pred_val[:, 2])
            
            # Update history
            history['train_loss'].append(train_loss.item())
            history['val_loss'].append(val_loss.item())
            history['env_rmse'].append(env_rmse)
            history['social_rmse'].append(social_rmse)
            history['gov_rmse'].append(gov_rmse)
            history['env_r2'].append(env_r2)
            history['social_r2'].append(social_r2)
            history['gov_r2'].append(gov_r2)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
                print(f"E: RMSE={env_rmse:.4f}, R²={env_r2:.4f} | "
                    f"S: RMSE={social_rmse:.4f}, R²={social_r2:.4f} | "
                    f"G: RMSE={gov_rmse:.4f}, R²={gov_r2:.4f}")
            
            # Check for early stopping
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                # Save best model
                best_model_state = {key: value.cpu() for key, value in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        return history
    
    # Save the model
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    # Load the model
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
    
    # Plot training history
    def plot_training_history(self, history):
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss
        axs[0, 0].plot(history['train_loss'], label='Train Loss')
        axs[0, 0].plot(history['val_loss'], label='Val Loss')
        axs[0, 0].set_title('Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('MSE Loss')
        axs[0, 0].legend()
        
        # Plot RMSE
        axs[0, 1].plot(history['env_rmse'], label='Environmental')
        axs[0, 1].plot(history['social_rmse'], label='Social')
        axs[0, 1].plot(history['gov_rmse'], label='Governance')
        axs[0, 1].set_title('RMSE')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('RMSE')
        axs[0, 1].legend()
        
        # Plot R²
        axs[1, 0].plot(history['env_r2'], label='Environmental')
        axs[1, 0].plot(history['social_r2'], label='Social')
        axs[1, 0].plot(history['gov_r2'], label='Governance')
        axs[1, 0].set_title('R²')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('R²')
        axs[1, 0].legend()
        
        # Empty plot
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    # Predict ESG scores for a document using sentence embeddings
    def predict_document_scores(self, graph, document_id, model_path=None):
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
        
        # Get sentences from document
        sentences = graph.run("""
            MATCH (d:Document {filename: $doc_id})-[:CONTAINS]->(s:Sentence)
            RETURN s.id as id, s.text as text
        """, doc_id=document_id).data()
        
        if not sentences:
            print(f"No sentences found for document {document_id}")
            return None
        
        # Process sentences
        num_sentences = len(sentences)
        feature_dim = 64
        X = np.zeros((num_sentences, feature_dim))
        
        # Create features
        for i, s in enumerate(sentences):
            seed = hash(s['text']) % 10000
            np.random.seed(seed)
            
            # Create embedding (replace with proper embeddings in production)
            embedding = np.random.rand(feature_dim)
            
            # Add ESG signals
            if 'environmental' in s['text'].lower():
                embedding[:20] += 0.3
            if 'social' in s['text'].lower():
                embedding[20:40] += 0.3
            if 'governance' in s['text'].lower():
                embedding[40:] += 0.3
                
            X[i] = embedding
        
        # Create edge index (fully connected for simplicity)
        edge_index = []
        for i in range(num_sentences):
            for j in range(i+1, num_sentences):
                edge_index.append([i, j])
                edge_index.append([j, i])
        
        # Convert to tensors
        x = torch.FloatTensor(X)
        edge_index = torch.LongTensor(edge_index).t()
        
        # Make prediction
        with torch.no_grad():
            out = self.model(x.to(self.device), edge_index.to(self.device))
            
        # Average scores across all sentences
        avg_scores = out.mean(dim=0).cpu().numpy()
        
        # Return results
        return {
            'environmental_score': float(avg_scores[0]),
            'social_score': float(avg_scores[1]),
            'governance_score': float(avg_scores[2])
        }


# Main function to train ESG GNN model
def main():
    # Connect to Neo4j
    NEO4J_URI = "bolt://localhost:7687"  # Update with your Neo4j URI
    NEO4J_USER = "neo4j"                  # Update with your username
    NEO4J_PASSWORD = "12345678"           # Update with your password
    
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print("Connected to Neo4j database")
    
    # Create dataset
    dataset = ESGGraphDataset(graph)
    data = dataset.get(0)
    
    print(f"Dataset created with {data.x.shape[0]} nodes and {data.edge_index.shape[1]} edges")
    
    # Create and train model
    model = ESGGNN(input_dim=data.x.shape[1])
    trainer = ESGGraphTrainer(model, device='cuda', lr=0.001)
    
    # Train the model
    history = trainer.train(data, epochs=300, patience=30)
    
    # Save the model
    trainer.save_model('esg_gnn_model.pt')
    
    # Plot training history
    trainer.plot_training_history(history)
    
    # Predict scores for the Apple document
    # Get document filename from database
    doc_result = graph.run("MATCH (d:Document) RETURN d.filename as filename").data()
    if doc_result:
        document_id = doc_result[0]['filename']
        scores = trainer.predict_document_scores(graph, document_id)
        
        print(f"Predicted ESG scores for document {document_id}:")
        print(f"Environmental: {scores['environmental_score']:.4f}")
        print(f"Social: {scores['social_score']:.4f}")
        print(f"Governance: {scores['governance_score']:.4f}")
    else:
        print("No documents found in database")

    print("Training and evaluation complete!")

if __name__ == "__main__":
    main()