## 💊 DDI Prediction with Various Encoders and Classifiers
 
### 📌 Overview
This project addresses the task of **Drug-Drug Interaction (DDI) type classification** using a **graph-based machine learning approach**.  
In polypharmacy, unexpected interactions between drugs can reduce efficacy or cause severe adverse effects. Since experimental validation is costly and time-consuming, this work explores the use of **Graph Neural Network (GNN) encoders** combined with different classifiers to model and predict DDI types efficiently.  
  
### - Dataset
- Source: **DrugBank**  
- **572 drugs**, **37,269 interactions**, **65 interaction types**  
- Features derived from multiple similarity matrices:  
  - Molecular substructures  (`features_m1.txt`)
  - Target proteins          (`features_m2.txt`)
  - Biological pathways      (`features_m3.txt`)
  - Side effects             (`features_m4.txt`)

*Reference* [*Predicting drug–drug interactions with knowledge graph embeddings* (Scientific Reports, 2022)](https://www.nature.com/articles/s41598-022-19999-4)  

### - Methodology
<p align="center">
  <img src="https://github.com/user-attachments/assets/fd5b2a9f-77db-4a30-aeac-ff61129a1fc2" width="400"/>
  <img src="https://github.com/user-attachments/assets/ae6223fd-c58e-4d64-87dd-4799abd5a947" width="400"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/354c17c5-c242-4932-8b77-ca6ed5fa6cf8" width="400"/>
  <img src="https://github.com/user-attachments/assets/7f891235-a3c8-47eb-8dfd-abd719cf41c2" width="400"/>
</p>

1. **Graph Construction**  
   - Nodes: drugs, Edges: interactions (labeled with 65 types)  
2. **Node Embedding (Phase 1)**  
   - **aprGCN**: Attributed PageRank + GCN (single graph)  
   - **bprGCN**: Binary Personalized PageRank + GCN (multi-graph)  
   - Baselines: GCN, GAT, GraphSAGE, GIN  
3. **Edge Classification (Phase 2)**  
   - Classifiers: **MLP**, **K-way predictor**, and attention-augmented variants  
   - Edge representation: concatenation of node embeddings **[h(u) ⊕ h(v)]**  

### - Results
- **bprGCN + MLP** improved over standard GCN + MLP  
- **GAT + MLP** achieved the best overall accuracy  
- Key insights:  
  - Multi-graph representations capture interaction patterns more effectively  
  - PageRank-based weighting improves embedding quality  
  - Attention layers provide additional performance gains  

<p align="left">
  <img src="https://github.com/user-attachments/assets/1e545601-3710-426c-b97e-05719f3e4b4a" width="800"/>
</p>

### 🚀 Conclusion
The study demonstrates that combining tailored **GNN encoders with suitable classifiers** can significantly improve DDI type prediction.  
The findings suggest that **multi-graph structures and PageRank-based embeddings** are particularly effective.  
Future work will focus on improving **explainability** to support real-world pharmaceutical applications.    

### ⚡ How to Run
#### Phase 1 — Node Embedding
```bash
bash run_1_encoder.sh
```
- Calls `train_graph_v1.py`(single-graph) or `train_graph_v2.py`(multi-graph)
- Custom `GCNConv` in `custom_convs.py` integrates edge features into aggregation

#### Phase 2 — Edge Classification
```bash
bash run_2_classifier.sh
```
- Runs MLP, K-way, Attention-MLP, and Attention-Kway classifiers
- Results saved in `./model/result/` 

#### File Structure
- `run_1_encoder.sh` / `run_2_classifier.sh` — batch scripts
- `custom_convs.py` — modified GCNConv with edge features
- `train_graph_v1.py` / `train_graph_v2.py` — node embedding (single / multi-graph)
- `train_fc_v1.py` / `train_fc_v2.py` — MLP and Attention-MLP
- `train_kway_v1.py` / `train_kway_v2.py` — K-way and Attention-Kway


*Team Project – KAIST CS471: Graph Machine Learning and Mining*
