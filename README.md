# Análise de Intervenção em Ansiedade com Visualizações Estáticas e Insights

Este repositório contém um notebook Python para análise detalhada de um conjunto de dados sobre intervenção em ansiedade, utilizando visualizações estáticas e técnicas de machine learning interpretável.

## Visão Geral

O notebook implementa uma análise abrangente de um conjunto de dados sintético de estudo de intervenção em ansiedade, focando na criação de visualizações estáticas utilizando Matplotlib e Seaborn, análise estatística e geração de insights. O fluxo de trabalho inclui análise exploratória completa, cálculo de valores SHAP para importância de características, visualizações estáticas avançadas e relatório automatizado de insights baseado em análise estatística.

## Funcionalidades Principais

- **Carregamento e Validação de Dados**: Pipeline completo de validação e processamento
- **Pré-processamento Robusto**: One-hot encoding e escalonamento de features
- **Interpretabilidade de Modelo**: Análise de valores SHAP para explicabilidade
- **Visualizações Estáticas Avançadas**:
  - Gráficos KDE para distribuições de ansiedade
  - Gráficos de violino para comparação entre grupos
  - Gráficos de coordenadas paralelas para trajetórias individuais
  - Visualização de hipergrafo utilizando NetworkX
- **Análise Estatística Rigorosa**: Bootstrap para intervalos de confiança
- **Relatório Automatizado**: Geração de insights baseada em achados estatísticos

## Requisitos Técnicos

- Python 3.x
- Dependências principais:
  - pandas
  - matplotlib
  - seaborn
  - networkx
  - shap
  - scikit-learn
  - numpy
  - scipy

## Estrutura do Fluxo de Trabalho

1. **Carregamento e Validação de Dados**
   - Verificação de colunas obrigatórias
   - Validação de tipos de dados e valores permitidos
   - Tratamento de identificadores duplicados

2. **Pré-processamento de Dados**
   - Preservação dos dados originais para referência
   - One-hot encoding das variáveis categóricas (grupo)
   - Escalonamento das características numéricas

3. **Análise SHAP**
   - Treinamento de modelo RandomForest para análise
   - Cálculo de valores SHAP para interpretabilidade
   - Visualização em formato de resumo

4. **Visualizações Estatísticas**
   - Implementação de múltiplas técnicas de visualização
   - Utilização de paleta de cores consistente
   - Tratamento de erros para robustez

5. **Análise Estatística**
   - Análise por grupo de intervenção
   - Cálculo de mudanças percentuais pré/pós-intervenção
   - Bootstrap com 500 reamostragens para intervalos de confiança

6. **Geração de Relatório**
   - Criação automática de insights baseados em análise estatística
   - Formatação em Markdown para fácil integração

## Compatibilidade de Ambiente

- Funciona em ambiente local ou Google Colab
- Detecção automática de ambiente para ajuste de caminhos
- Gerenciamento de diretório de saída configurável

## Artefatos Gerados

O notebook gera os seguintes arquivos:

- `shap_summary.png`: Visualização da importância das características
- `kde_plot.png`: Distribuições de ansiedade pré e pós-intervenção
- `violin_plot.png`: Comparação da ansiedade pós-intervenção entre grupos
- `parallel_coordinates_plot.png`: Trajetórias individuais pré/pós-intervenção
- `hypergraph.png`: Rede de relações entre participantes e níveis de ansiedade
- `summary.txt`: Estatísticas descritivas e intervalos de confiança
- `insights_report.md`: Relatório detalhado com análise e recomendações

## Como Executar

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/anxiety-intervention-analysis.git
cd anxiety-intervention-analysis

# Instalar dependências
pip install -r requirements.txt

# Executar o notebook
jupyter notebook io.ipynb
```

## Considerações de Implementação

- Tratamento robusto de erros em todas as funções
- Documentação detalhada de código com docstrings
- Configurações personalizáveis via constantes no início do notebook
- Implementação adaptada para conjuntos de dados pequenos e médios

## Limitações e Trabalhos Futuros

- Atualmente utiliza um conjunto de dados sintético para demonstração
- Potencial para expansão com técnicas de modelagem mais avançadas
- Possibilidade de implementação de visualizações interativas em versões futuras

## Autor
Hélio Craveiro Pessoa Júnior
