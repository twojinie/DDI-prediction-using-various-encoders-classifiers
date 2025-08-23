## 💊 DDI-prediction-using-various-encoders-classifiers
 
### 📌 프로젝트 개요
이 프로젝트는 **약물-약물 상호작용(Drug-Drug Interaction, DDI) 유형을 예측**하는 것을 목표로 합니다.  
DDI는 다약제 치료(polypharmacy) 상황에서 약물 간 상호작용이 발생해 약효가 떨어지거나 심각한 부작용이 나타나는 문제로, 환자 안전과 신약 개발 과정에서 매우 중요한 과제입니다.  
전통적인 in-vitro / in-vivo 실험은 비용과 시간이 많이 들기 때문에, **그래프 신경망(Graph Neural Networks, GNNs)** 을 이용한 계산적 접근 방식을 시도했습니다. 

### 🗂️ 데이터셋
- 출처: **DrugBank**  
- **572개 약물**, **37,269개 상호작용(edge)**, **65가지 상호작용 유형(label)**  
- 각 약물 쌍(edge)에 대해 다양한 **similarity matrix**를 기반으로 특징을 추출  
  - 약물 구조(Substructure)  
  - 타겟 단백질(Target)  
  - 생물학적 경로(Pathway)  
  - 부작용 정보(Side effect)

*참고 논문* [*Predicting drug–drug interactions with knowledge graph embeddings* (Scientific Reports, 2022)](https://www.nature.com/articles/s41598-022-19999-4)  


### 🔍 방법론
<p align="center">
  <img src="https://github.com/user-attachments/assets/ae6223fd-c58e-4d64-87dd-4799abd5a947" width="300"/>
  <img src="https://github.com/user-attachments/assets/354c17c5-c242-4932-8b77-ca6ed5fa6cf8" width="300"/>
  <img src="https://github.com/user-attachments/assets/7f891235-a3c8-47eb-8dfd-abd719cf41c2" width="300"/>
</p>

1. **그래프 구성**  
   - 노드 = 약물  
   - 엣지 = 상호작용 (interaction type을 라벨로 가짐)  

2. **인코더 (노드 임베딩)**  
   - **aprGCN**: 단일 그래프 기반 Attributed PageRank GCN  
   - **bprGCN**: 상호작용 유형별로 그래프를 나눈 Binary PageRank GCN  
   - PageRank 기반 가중치와 GNN 임베딩을 결합  

3. **분류기 (Classifier)**  
   - **MLP (다층 퍼셉트론)**  
   - **K-way predictor**  
   - 각 모델에 **Self-Attention**을 추가한 변형도 실험

   
### 📊 성능 결과
- 다양한 **인코더–분류기 조합**을 실험한 결과:  
- **bprGCN + MLP** 조합이 가장 좋은 성능을 보였으며, 기본 GCN+MLP보다 우수  
- 핵심 발견:  
  - 상호작용 유형별로 분리한 **멀티 그래프 표현**이 더 정확한 예측에 도움  
  - PageRank 기반 가중치가 임베딩 품질 향상에 기여  
  - Self-Attention을 통해 추가적인 성능 개선 가능성 확인  

### 🚀 결론 및 성과
- **새로운 인코더–분류기 조합**을 통해 DDI 유형 분류 가능성을 입증  
- **멀티 그래프 + PageRank + GNN** 조합이 상호작용 패턴을 효과적으로 포착  
- 앞으로의 확장 방향:  
  - 실제 제약 분야 활용을 위한 **설명 가능성(Explainability)** 강화  

### ✔️ 실행 방법

```bash
# 환경 세팅
pip install -r requirements.txt

# 학습 실행 (예시)
python main.py --encoder bprGCN --classifier MLP
```

*팀 프로젝트 (KAIST CS471: Graph Machine Learning and Mining 수업)*
