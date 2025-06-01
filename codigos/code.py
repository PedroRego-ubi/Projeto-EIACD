import pandas as pd
import chardet
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import zscore, norm

# Configuração de logging para registrar operações e erros
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='../analise_dados.log',
    filemode='w'
)

def debug_file_structure(file_path):
    """Detecta encoding e lê arquivo CSV, lidando com diferentes formatos"""
    try:
        with open(file_path, 'rb') as f:
            rawdata = f.read(100000)
            encoding = chardet.detect(rawdata)['encoding']

        # Tenta ler com encoding detectado, depois tenta utf-8 como fallback
        try:
            df = pd.read_csv(file_path, encoding=encoding, header=None)
        except:
            df = pd.read_csv(file_path, encoding='utf-8', header=None)

        return df
    except Exception as e:
        logging.error(f"Erro ao ler {file_path}: {str(e)}")
        return None

def reader(file_path):
    """Lê arquivo CSV, filtra por municípios e padroniza estrutura"""
    try:
        with open(file_path, 'rb') as f:
            rawdata = f.read(10000)
            encoding = chardet.detect(rawdata)['encoding']

        with open(file_path, 'r', encoding=encoding) as f:
            linhas = [linha.strip() for linha in f if linha.strip()]

        if not linhas:
            return pd.DataFrame()

        cabecalhos = [col.strip() for col in linhas[0].split(',')]
        dados = []
        for linha in linhas[1:]:
            valores = [v.strip() for v in linha.split(',')]
            if len(valores) != len(cabecalhos):
                valores = valores + [''] * (len(cabecalhos) - len(valores))
            dados.append(valores[:len(cabecalhos)])

        df = pd.DataFrame(dados, columns=cabecalhos)
        pd.set_option('display.max_columns', None)

        # Filtra apenas dados municipais
        col_geo = next((col for col in df.columns if 'Âmbito Geográfico' in col), None)
        if col_geo:
            df = df[df[col_geo].str.contains('Município', na=False)]

        return df
    except Exception as e:
        logging.error(f"Erro no reader: {str(e)}")
        return pd.DataFrame()

def processar_csv(file_path, nome_metrica):
    """Processa arquivo CSV e extrai colunas relevantes (município, ano, valor)"""
    try:
        logging.info(f"Iniciando processamento de: {file_path}")
        df_raw = debug_file_structure(file_path)

        if df_raw is None or df_raw.empty:
            logging.warning(f"Arquivo vazio ou não lido corretamente: {file_path}")
            return pd.DataFrame()

        header_row = df_raw.iloc[0].astype(str)

        # Identifica colunas relevantes
        col_municipio = None
        col_ano = None
        col_valor = None
        col_tipo = None

        for i, col_nome in enumerate(header_row):
            nome = col_nome.strip()
            if nome.endswith("Nome Região (Portugal)"):
                col_municipio = i
            elif "Ano" in nome or "ano" in nome:
                col_ano = i
            elif "Valor" in nome or "valor" in nome or "pessoas" in nome.lower():
                col_valor = i
            elif "Âmbito Geográfico" in nome or "Município" in nome:
                col_tipo = i

        if None in [col_municipio, col_ano, col_valor]:
            logging.warning(f"Colunas essenciais não encontradas em {file_path}")
            return pd.DataFrame()

        df_raw = df_raw.drop(index=0).reset_index(drop=True)

        # Filtra apenas dados totais (remove divisão por gênero)
        gender_col_index = None
        for i in range(df_raw.shape[1]):
            sample_values = df_raw[i].astype(str).str.lower().unique()
            if any(val in sample_values for val in ['homens', 'mulheres', 'total']):
                gender_col_index = i
                break

        if gender_col_index is not None:
            df_raw = df_raw[df_raw[gender_col_index].astype(str).str.lower().str.strip() == "total"]

        # Cria DataFrame padronizado
        df_processed = pd.DataFrame({
            'municipios': df_raw[col_municipio].astype(str).str.strip(),
            'ano': df_raw[col_ano],
            nome_metrica + '_valor': df_raw[col_valor]
        })

        # Filtra por municípios se a coluna de tipo existir
        if col_tipo is not None:
            mask = df_raw[col_tipo].astype(str).str.contains('Município', case=False, na=False)
            df_processed = df_processed[mask]

        # Converte para valores numéricos
        df_processed['ano'] = pd.to_numeric(df_processed['ano'], errors='coerce')
        df_processed[nome_metrica + '_valor'] = pd.to_numeric(
            df_processed[nome_metrica + '_valor'], errors='coerce')

        resultado = df_processed.dropna()
        logging.info(f"Processamento concluído: {file_path} - {len(resultado)} registros válidos")
        return resultado

    except Exception as e:
        logging.error(f"Erro ao processar {file_path}: {str(e)}")
        return pd.DataFrame()

def filtrar_por_iqr(df, coluna_valor):
    """Remove outliers usando o método do Intervalo Interquartil (IQR)"""
    try:
        logging.info(f"Iniciando filtro IQR na coluna '{coluna_valor}'")

        if df.empty:
            logging.warning("DataFrame vazio - nada para filtrar")
            return df

        if coluna_valor not in df.columns:
            logging.warning(f"Coluna '{coluna_valor}' não encontrada")
            return df

        # Remove valores inválidos e converte para numérico
        df = df.dropna(subset=[coluna_valor])
        df[coluna_valor] = pd.to_numeric(df[coluna_valor], errors='coerce')
        df = df.dropna(subset=[coluna_valor])

        # Calcula quartis e IQR
        Q1 = df[coluna_valor].quantile(0.25)
        Q3 = df[coluna_valor].quantile(0.75)
        IQR = Q3 - Q1

        # Define limites
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filtra os dados
        filtered_df = df[(df[coluna_valor] >= lower_bound) & (df[coluna_valor] <= upper_bound)].copy()

        logging.info(f"IQR aplicado: {len(df)} → {len(filtered_df)} registros")
        return filtered_df

    except Exception as e:
        logging.error(f"Erro ao filtrar por IQR: {str(e)}")
        return df

def filtrar_por_zscore(df, coluna_valor, limiar=3):
    """Remove outliers usando Z-Score (desvios padrão da média)"""
    try:
        logging.info(f"Iniciando filtro Z-Score na coluna '{coluna_valor}' com limiar ±{limiar}")

        if df.empty:
            logging.warning("DataFrame vazio - nada para filtrar")
            return df

        if coluna_valor not in df.columns:
            logging.warning(f"Coluna '{coluna_valor}' não encontrada")
            return df

        # Remove valores inválidos e converte para numérico
        df = df.dropna(subset=[coluna_valor])
        df[coluna_valor] = pd.to_numeric(df[coluna_valor], errors='coerce')
        df = df.dropna(subset=[coluna_valor])

        # Calcula Z-Scores e filtra
        z_scores = zscore(df[coluna_valor])
        filtered_df = df[(abs(z_scores) < limiar)].copy()

        logging.info(f"Z-Score aplicado: {len(df)} → {len(filtered_df)} registros")
        return filtered_df

    except Exception as e:
        logging.error(f"Erro ao filtrar por Z-Score: {str(e)}")
        return df

def merge_dataframes(df1, df2, merge_on=['municipios', 'ano']):
    """Combina dois DataFrames mantendo apenas registros presentes em ambos"""
    try:
        logging.info(f"Iniciando merge com colunas: {merge_on}")

        if df1.empty or df2.empty:
            logging.warning("Um dos DataFrames está vazio")
            return pd.DataFrame()

        # Verifica se colunas para merge existem
        missing_cols = [col for col in merge_on if col not in df1.columns or col not in df2.columns]
        if missing_cols:
            logging.warning(f"Colunas faltando para merge: {missing_cols}")
            return pd.DataFrame()

        merged = pd.merge(df1, df2, on=merge_on, how='inner')
        logging.info(f"Merge concluído com {len(merged)} registros")
        return merged

    except Exception as e:
        logging.error(f"Erro ao fazer merge: {str(e)}")
        return pd.DataFrame()

def carregar_e_processar_todos():
    """Carrega e processa todos os arquivos de dados da pasta Data"""
    arquivos_metricas = {
        'densidade': 'Data/densidade_populacional.csv',
        'emprego': 'Data/emprego_por_conta_de_outrem.csv',
        'mortalidade': 'Data/taxa_bruta_de_mortalidade.csv',
        'natalidade': 'Data/taxa_bruta_de_natalidade.csv'
    }

    dfs = {}
    for nome, caminho in arquivos_metricas.items():
        df = processar_csv(caminho, nome)
        dfs[nome] = df

    return dfs

def criar_matriz_dados_df(df, x_col, y_col):
    """Executa análise de clustering hierárquico e exibe gráfico"""
    # Padroniza nomes de colunas
    df.columns = df.columns.str.strip().str.lower()
    x_col = x_col.lower()
    y_col = y_col.lower()

    # Verifica colunas necessárias
    if not {'municipios', 'ano', x_col, y_col}.issubset(df.columns):
        print(f"Erro: DataFrame precisa das colunas 'municipios', 'ano', '{x_col}' e '{y_col}'")
        return

    # Seleção interativa do ano
    anos_disponiveis = sorted(df['ano'].unique())
    while True:
        try:
            print(f"\nAnos disponíveis: {', '.join(map(str, anos_disponiveis))}")
            ano = int(input("Digite o ano para análise: "))
            if ano in anos_disponiveis:
                break
            print(f"Ano {ano} não disponível. Tente novamente.")
        except ValueError:
            print("Digite um número válido.")

    # Filtra dados para o ano selecionado
    df_filtrado = df[df["ano"] == ano]

    # Prepara dados para clustering
    v1 = df_filtrado[x_col].tolist()
    v2 = df_filtrado[y_col].tolist()
    municipios = df_filtrado["municipios"].tolist()

    # Remove valores faltantes
    matriz = [(x, y) for x, y in zip(v1, v2) if pd.notna(x) and pd.notna(y)]
    municipios = [m for m, x, y in zip(municipios, v1, v2) if pd.notna(x) and pd.notna(y)]

    if not matriz:
        print("Nenhum dado válido para este ano.")
        return

    # Normaliza dados
    matriz_np = np.array(matriz)
    matriz_normalizada = StandardScaler().fit_transform(matriz_np)

    # Executa clustering
    modelo = AgglomerativeClustering(n_clusters=4, linkage='ward')
    labels = modelo.fit_predict(matriz_normalizada)

    # Configura gráfico
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(matriz_np[:, 0], matriz_np[:, 1], c=labels, cmap='viridis', s=100, alpha=0.7)
    plt.xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    plt.title(f"Análise de Clusters - {x_col.replace('_', ' ')} vs {y_col.replace('_', ' ')}\nAno {ano}", pad=20)

    # Adiciona legenda com municípios
    for i, txt in enumerate(municipios):
        plt.annotate(txt, (matriz_np[i, 0], matriz_np[i, 1]), fontsize=8, alpha=0.7)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.show()

    # Exibe resultados por município
    print("\nClassificação por município:")
    for municipio, cluster in zip(municipios, labels):
        print(f"{municipio}: Cluster {cluster}")

def criar_histograma(df, nome_dataset, coluna_valor):
    """Gera histograma interativo para análise de distribuição por município"""
    # Verifica colunas necessárias
    if not {'municipios', 'ano', coluna_valor}.issubset(df.columns):
        print(f"Erro: Faltam colunas necessárias ('municipios', 'ano', '{coluna_valor}')")
        return

    # Converte valores para numérico
    df[coluna_valor] = pd.to_numeric(df[coluna_valor], errors='coerce')
    df = df.dropna(subset=[coluna_valor, 'ano'])

    # Seleciona ano interativamente
    anos_disponiveis = sorted(df['ano'].unique())
    while True:
        try:
            print(f"\nAnos disponíveis: {', '.join(map(str, anos_disponiveis))}")
            ano = int(input("Digite o ano para o histograma: "))
            if ano in anos_disponiveis:
                break
            print(f"Ano {ano} não encontrado. Tente novamente.")
        except ValueError:
            print("Digite um número válido.")

    # Filtra por ano e ordena
    df_ano = df[df['ano'] == ano].sort_values('municipios')

    if df_ano.empty:
        print("Nenhum dado disponível para este ano.")
        return

    # Configuração do gráfico
    plt.figure(figsize=(14, 7))
    bars = plt.bar(df_ano['municipios'], df_ano[coluna_valor], color='#1f77b4', alpha=0.7)

    # Linha de média e mediana
    media = df_ano[coluna_valor].mean()
    mediana = df_ano[coluna_valor].median()

    plt.axhline(media, color='red', linestyle='--', linewidth=1.5, label=f'Média: {media:,.2f}')
    plt.axhline(mediana, color='green', linestyle='-.', linewidth=1.5, label=f'Mediana: {mediana:,.2f}')

    # Customização do gráfico
    plt.title(f"Distribuição de {nome_dataset.replace('_', ' ').title()} por Município - {ano}", pad=20, fontsize=14)
    plt.xlabel('Municípios', fontsize=12)
    plt.ylabel(nome_dataset.replace('_', ' ').title(), fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Formatação do eixo Y
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # Ajusta layout e exibe
    plt.tight_layout()
    plt.show()

def mostrar_distribuicao_normal(dfs):
    """Compara distribuição dos dados com distribuição normal teórica"""
    print("\nDatasets disponíveis:")
    for i, nome in enumerate(dfs.keys(), 1):
        print(f"[{i}] {nome.capitalize()}")

    try:
        escolha = int(input("\nEscolha o dataset para visualizar: ").strip()) - 1
        datasets = list(dfs.keys())

        if 0 <= escolha < len(datasets):
            nome_dataset = datasets[escolha]
            df = dfs[nome_dataset]
            coluna = f"{nome_dataset}_valor"

            if df.empty or coluna not in df.columns:
                print("Dataset inválido ou coluna não encontrada.")
                return

            # Prepara dados
            dados = pd.to_numeric(df[coluna], errors='coerce').dropna()

            if dados.empty:
                print("Sem dados numéricos válidos para exibir.")
                return

            # Ajusta distribuição normal
            media, desvio = norm.fit(dados)

            # Configura gráfico
            plt.figure(figsize=(10, 6))

            # Histograma dos dados reais
            plt.hist(dados, bins=30, density=True, alpha=0.6,
                     color='skyblue', edgecolor='black', label='Dados Reais')

            # Curva normal teórica
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, media, desvio)
            plt.plot(x, p, 'r', linewidth=2, label='Distribuição Normal')

            # Customização
            plt.title(f'Comparação com Distribuição Normal\n{nome_dataset.replace("_", " ").title()}', pad=20)
            plt.xlabel('Valores', fontsize=12)
            plt.ylabel('Densidade', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.tight_layout()
            plt.show()

            # Exibe estatísticas
            print(f"\nEstatísticas para {nome_dataset.replace('_', ' ')}:")
            print(f"- Média: {media:.2f}")
            print(f"- Desvio Padrão: {desvio:.2f}")
            print(f"- Assimetria: {dados.skew():.2f}")
            print(f"- Curtose: {dados.kurtosis():.2f}")

        else:
            print("Opção inválida.")

    except ValueError:
        print("Entrada inválida.")