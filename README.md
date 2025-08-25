## 💊 DDI-prediction-using-various-encoders-classifiers
 
### 📌 프로젝트 개요
이 프로젝트는 **약물-약물 상호작용(Drug-Drug Interaction, DDI) 유형 분류** 문제를 **그래프 머신러닝(Graph Machine Learning)** 접근으로 해결하고자 합니다.  
DDI는 다약제 치료(polypharmacy) 상황에서 약물 간 상호작용이 발생해 약효가 떨어지거나 심각한 부작용이 나타나는 문제로, 환자 안전과 신약 개발 과정에서 매우 중요한 과제입니다.  
전통적인 in-vitro / in-vivo 실험적 검증에는 높은 비용과 시간이 소요되기 때문에 **그래프 신경망(Graph Neural Networks, GNNs) 인코더**와 **다양한 분류기(Classifier)** 조합을 실험하여, 대규모 복잡한 관계 데이터를 효율적으로 모델링하는 방법을 탐구했습니다.

### 🗂️ 데이터셋
- 출처: **DrugBank**  
- **572개 약물**, **37,269개 상호작용(edge)**, **65가지 상호작용 유형(label)**  
- 각 약물 쌍(edge)에 대해 다양한 **similarity matrix**를 기반으로 특징을 추출  
  - 약물 구조(Substructure)  (`features_m1.txt`)
  - 타겟 단백질(Target)      (`features_m2.txt`)
  - 생물학적 경로(Pathway)   (`features_m3.txt`)
  - 부작용 정보(Side effect) (`features_m4.txt`)

*참고 논문* [*Predicting drug–drug interactions with knowledge graph embeddings* (Scientific Reports, 2022)](https://www.nature.com/articles/s41598-022-19999-4)  

### 🔍 방법론
<p align="center">
  <img src="https://github.com/user-attachments/assets/fd5b2a9f-77db-4a30-aeac-ff61129a1fc2" width="400"/>
  <img src="https://github.com/user-attachments/assets/ae6223fd-c58e-4d64-87dd-4799abd5a947" width="400"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/354c17c5-c242-4932-8b77-ca6ed5fa6cf8" width="400"/>
  <img src="https://github.com/user-attachments/assets/7f891235-a3c8-47eb-8dfd-abd719cf41c2" width="400"/>
</p>

1) **그래프 구성**: 노드=약물, 엣지=DDI(라벨=65개 type)  
2) **노드 임베딩(Phase 1, Encoder)**:  
   - **aprGCN**(단일 그래프, Attributed PageRank + GCN)  
   - **bprGCN**(타입별 멀티그래프, Binary Personalized PageRank + GCN)  
   - GCN/GAT/GraphSAGE/GIN 등 여러 GNN 비교
3) **엣지 타입 분류(Phase 2, Classifier)**:  
   - **MLP**, **K‑way predictor**  
   - Self-Attention 결합 (Attention-MLP, Attention-Kway)
   - 엣지 표현: 노드 임베딩 concat **[h(u) ⊕ h(v)]**



### 📊 성능 결과
- 다양한 **인코더–분류기 조합**을 실험 
- **bprGCN + MLP**가 vanilla GCN + MLP 대비 일관된 성능 향상  
- 전체 비교에서는 **GAT + MLP** 조합이 최고 성능  
- 주요 발견:  
  - 상호작용 유형별로 분리한 **멀티 그래프 표현**이 패턴 학습에 효과적  
  - PageRank 기반 가중치가 임베딩 품질을 높임  
  - Self-Attention이 추가적인 성능 향상 가능성을 제공  

<p align="left">
  <img src="https://github.com/user-attachments/assets/1e545601-3710-426c-b97e-05719f3e4b4a" width="800"/>
</p>

### 🚀 결론 및 성과
- **새로운 인코더–분류기 조합**을 통해 DDI 유형 분류 가능성을 입증  
- **멀티 그래프 + PageRank + GNN** 조합이 상호작용 패턴을 효과적으로 포착  
- 앞으로의 확장 방향:  
  - 실제 제약 분야 활용을 위한 **설명 가능성(Explainability)** 강화  

### ✔️ 실행 방법
#### Phase 1 — 노드 임베딩 학습
```bash
bash run_1_encoder.sh
```
- 내부적으로 `train_graph_v1.py`(단일 그래프) / `train_graph_v2.py`(멀티그래프)를 호출
- `custom_convs.py`: PyG `GCNConv` 수정본. **엣지 특성(edge feature)** 을 이웃 집계에 반영하도록 확장

#### Phase 2 — 엣지 타입 분류(65‑class)
```bash
bash run_2_classifier.sh
```
- 네 분류기 순차 실행: **MLP**, **K‑way**, **Attention‑MLP**, **Attention‑K‑way**
- 결과는 `./model/result/` 에 저장

#### 파일 설명
- `run_1_encoder.sh` / `run_2_classifier.sh` — Phase 1/2 일괄 실행
- `custom_convs.py` — Edge feature를 집계에 반영하는 커스텀 GCNConv
- `train_graph_v1.py` / `train_graph_v2.py` — 노드 임베딩 학습(단일/멀티그래프)
- `train_fc_v1.py` / `train_fc_v2.py` — MLP·Attention‑MLP 분류
- `train_kway_v1.py` / `train_kway_v2.py` — K‑way·Attention‑K‑way 분류


*팀 프로젝트 (KAIST CS471: Graph Machine Learning and Mining 수업)*
