# Projeto-EIACD

Certifique-se que os arquivos .csv estão na pasta Data/ com os nomes corretos:

densidade_populacional.csv

emprego_por_conta_de_outrem.csv

taxa_bruta_de_natalidade.csv

taxa_bruta_de_mortalidade.csv

Execute o programa:

python main.py
📌 Funcionalidades
Menu Principal
[1-5] Análises cruzadas entre indicadores (ex: densidade × emprego)

[6] Filtrar outliers usando IQR

[7] Filtrar outliers usando Z-Score

[8] Gerar histograma

[9] Mostrar distribuição normal

[20] Vai ter que testar pra descobrir

[0] Sair

Exemplos de Análises
Geração de gráfico de clusters por município com base em dois indicadores

Visualização de histogramas por ano

Remoção de valores extremos que podem distorcer as análises

📈 Técnicas Utilizadas
Clustering hierárquico (Agglomerative)

Padronização de dados (StandardScaler)

Detecção e remoção de outliers (IQR e Z-Score)

Visualização com Matplotlib

# Fontes de Dados
Os dados devem conter colunas com:

Nome do município

Ano

Valor do indicador

Âmbito geográfico (para filtragem de municípios)

# Observações
O programa foi testado com dados do INE (Instituto Nacional de Estatística).

Os logs são gerados automaticamente no arquivo analise_dados.log.

# Agradecimentos
Este projeto foi desenvolvido como parte da unidade curricular Elementos de Inteligência Artificial e Ciência de Dados da Universidade da Beira Interior.

# Links:

https://docs.google.com/document/d/14UL4D91vhZYDywp9MNmAjEIin1LbzWiVJkge_Q7AwSc/edit?tab=t.0

https://docs.google.com/presentation/d/1N8vPq0K2W1crCcsekbvOGgNki9webzCLhQialGjTPXc/edit?slide=id.g35f83489e09_0_63#slide=id.g35f83489e09_0_63
