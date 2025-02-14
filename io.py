import pandas as pd
import numpy as np
import os
from google.colab import drive

# Montar o Google Drive
drive.mount('/content/drive')

# Definir caminhos dos arquivos
input_url = 'http://docs.google.com/spreadsheets/d/1q4s8zkiODrE0NrgRYn2iG7RsAJk5gSJXj-U854ujcFY/export?format=csv'
output_file = '/content/drive/MyDrive/data_filtrado.csv'

# Parâmetros
keywords = ["SEM", "SEM Graph", "MASEM", "LSEM", "RAG", "equation modeling", "SEMGraph", "MA-SEM", "LSEM", "Retriveal"]
proximity_threshold = 4

# Função para verificar a proximidade das palavras-chave
def test_keyword_proximity(cell_values, keywords, threshold):
    filtered_cell_values = [value for value in cell_values if isinstance(value, str) and value.strip()]
    keyword_positions = []
    for i, cell in enumerate(filtered_cell_values):
        for keyword in keywords:
            if keyword.lower() in cell.lower():
                keyword_positions.append(i)
    # Verifica se há pelo menos duas palavras-chave próximas
    for i in range(len(keyword_positions)):
        for j in range(i + 1, len(keyword_positions)):
            if abs(keyword_positions[i] - keyword_positions[j]) <= threshold:
                return True
    return False

# Função para ler o arquivo CSV
def read_csv_file(file_url):
    try:
        return pd.read_csv(file_url)
    except Exception as e:
        raise ValueError(f"Erro ao ler o arquivo CSV: {e}")

# Função para filtrar os dados
def filter_data(data, keywords, threshold):
    filtered_data = []
    total_rows = len(data)
    for i, (_, row) in enumerate(data.iterrows()):
        cell_values = [str(value) for value in row.values if isinstance(value, str) and value.strip()]
        if not cell_values:
            continue  # Ignorar linhas sem conteúdo
        if test_keyword_proximity(cell_values, keywords, threshold):
            filtered_data.append(row)
        print(f"\rFiltrando dados: Processando linha {i + 1} de {total_rows}", end="")
    print("\n")
    return pd.DataFrame(filtered_data)

# Função para exportar os dados filtrados
def export_filtered_data(filtered_data, file_path):
    try:
        filtered_data.to_csv(file_path, index=False)
        print(f"Dados filtrados exportados para: {file_path}")
    except Exception as e:
        raise ValueError(f"Erro ao exportar o arquivo CSV: {e}")

# Fluxo principal do script
try:
    # Ler o arquivo CSV
    data = read_csv_file(input_url)
    # Filtrar os dados
    filtered_data = filter_data(data, keywords, proximity_threshold)
    # Exportar os dados filtrados
    export_filtered_data(filtered_data, output_file)
except Exception as e:
    print(f"Erro: {e}")