# Aprimoramento Experimental do DistilBERT em Tarefas de Classificação (TPF-02)

Este repositório contém o ambiente experimental desenvolvido para a disciplina de **Redes Neurais**. O projeto investiga os limites de generalização e resiliência da arquitetura comprimida **DistilBERT** ao submetê-la a modificações estruturais em duas frentes complementares: injeção de ruído nos dados de entrada e introdução de forte regularização arquitetural.

## Estrutura das Modificações (TPF-02)

Diferente da primeira etapa (TPF-01), que avaliou apenas a inferência com pesos pré-treinados, nesta segunda etapa foi realizado o *fine-tuning* completo do zero aplicando:
1. **Dimensão dos Dados:** *Data Augmentation* estático via WordNet (substituição aleatória de 10% dos tokens por sinônimos).
2. **Dimensão Arquitetural:** Nova cabeça de classificação baseada em uma rede MLP Customizada em PyTorch ($768 \times 768 \rightarrow \text{ReLU} \rightarrow 768 \times 256 \rightarrow \text{ReLU} \rightarrow 2
\text{ saídas}$).
3. **Regularização:** Inserção de *Dropout* agressivo de 40% e 20% nas camadas internas da MLP para conter o *overfitting*.
4. **Otimização:** Esquema de treino monitorado por *Cosine Annealing with Restarts*, *Warmup Steps* e *Early Stopping*.

---

## Instruções de Execução no Google Colab

Para reproduzir os experimentos e coletar as métricas sem quebras de escopo, siga estritamente a ordem de execução abaixo:

### 1. Preparação do Ambiente (Dependências)
Antes de iniciar qualquer treinamento, certifique-se de executar **todas as células de instalação e carregamento de dependências**, englobando os requisitos do TPF-01 e do TPF-02. Isso garante a correta instalação de bibliotecas essenciais como `transformers`, `datasets`, `nlpaug` e o isolamento dos pacotes linguísticos do `nltk`.

### 2. Pipeline de Data Augmentation
Execute a célula intitulada **"O código de Aumento de Dados (Data Augmentation)"**. Ela aplicará as perturbações lexicais baseadas no WordNet sobre as sentenças do dataset SST-2 e estruturará os tensores na memória RAM.

### 3. Ajuste Fino Customizado
Para disparar o treinamento final do modelo com a nova topologia, execute sequencialmente as células posicionadas abaixo da divisão:
`### Experimento Final: Arquitetura Customizada (MLP + Dropout) e Early Stopping`

> **Nota de Infraestrutura:** O processo foi otimizado utilizando técnicas de *Mixed Precision* (`fp16=True`) e Acúmulo de Gradientes no ambiente de GPU dedicada (NVIDIA T4). Certifique-se de manter o acelerador de hardware ativado nas configurações do notebook.

---

## Resultados Obtidos (SST-2)

* **Modelo pré-treinado (TPF-01):** 91.06% de acurácia (linha de referência).
* **Modelo Customizado (TPF-02):** **90.13% de acurácia** máxima atingida na Época 3.
