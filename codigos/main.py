from codigos import code
import time
import os
import platform

def limpar_tela():
    """Limpa a tela do console de forma compatível com diferentes sistemas operacionais"""
    time.sleep(1)
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')

def mostrar_menu():
    print('''\nESCOLHA A ANÁLISE:
[1] Densidade × Emprego
[2] Emprego × Natalidade
[3] Emprego × Mortalidade
[4] Densidade × Mortalidade
[5] Densidade × Natalidade
[6] Filtrar dados (IQR)
[7] Filtrar dados (Z-Score)
[8] Gerar histograma
[9] Gerar distribuição normal
[20] O que isto faz??
[0] Sair''')

def processar_analise(dfs, opcao):
    """Processa as análises principais"""
    match opcao:
        case '1':
            df = code.merge_dataframes(dfs['densidade'], dfs['emprego'])
            return df, "Densidade vs Emprego", "densidade_valor", "emprego_valor"
        case '2':
            df = code.merge_dataframes(dfs['emprego'], dfs['natalidade'])
            return df, "Emprego vs Natalidade", "emprego_valor", "natalidade_valor"
        case '3':
            df = code.merge_dataframes(dfs['emprego'], dfs['mortalidade'])
            return df, "Emprego vs Mortalidade", "emprego_valor", "mortalidade_valor"
        case '4':
            df = code.merge_dataframes(dfs['densidade'], dfs['mortalidade'])
            return df, "Densidade vs Mortalidade", "densidade_valor", "mortalidade_valor"
        case '5':
            df = code.merge_dataframes(dfs['densidade'], dfs['natalidade'])
            return df, "Densidade vs Natalidade", "densidade_valor", "natalidade_valor"
        case _:
            return None, "", "", ""

def filtrar_dataset(dfs, metodo='iqr'):
    """Interface para filtrar qualquer dataset"""
    print("\nDatasets disponíveis:")
    for i, nome in enumerate(dfs.keys(), 1):
        print(f"[{i}] {nome.capitalize()}")

    escolha = input("\nEscolha o dataset para filtrar: ").strip()
    datasets = list(dfs.keys())

    try:
        idx = int(escolha) - 1
        if 0 <= idx < len(datasets):
            nome_dataset = datasets[idx]
            coluna_valor = f"{nome_dataset}_valor"

            if metodo == 'iqr':
                dfs[nome_dataset] = code.filtrar_por_iqr(dfs[nome_dataset], coluna_valor)
            else:
                limiar = float(input("Digite o limiar do Z-Score (padrão=3): ") or 3)
                dfs[nome_dataset] = code.filtrar_por_zscore(dfs[nome_dataset], coluna_valor, limiar)
        else:
            print("Opção inválida.")
    except ValueError:
        print("Digite um número válido.")

    return dfs

def main():
    # Carrega todos os dataframes
    dfs = code.carregar_e_processar_todos()

    # Verifica dados carregados
    if any(df.empty for df in dfs.values()):
        print("\nErro: Alguns arquivos não foram carregados corretamente.")
        print("Verifique se os arquivos estão na pasta 'Data'.")
        return

    while True:
        limpar_tela()
        mostrar_menu()
        escolha = input('\nOpção: ').strip()

        match escolha:
            case '1' | '2' | '3' | '4' | '5':
                df, titulo, x_col, y_col = processar_analise(dfs, escolha)
                if df is not None:
                    print(f"\n{titulo}:")
                    print(df.head())
                    code.criar_matriz_dados_df(df, x_col, y_col)

            case '6':
                dfs = filtrar_dataset(dfs, metodo='iqr')

            case '7':
                dfs = filtrar_dataset(dfs, metodo='zscore')
            case '8':
                print("\nDatasets disponíveis:")
                for i, (nome, df) in enumerate(dfs.items(), 1):
                    print(f"[{i}] {nome.replace('_', ' ').title()} ({len(df)} registros)")

                escolha = input("\nEscolha o dataset: ").strip()
                datasets = list(dfs.keys())

                try:
                    idx = int(escolha) - 1
                    if 0 <= idx < len(datasets):
                        nome = datasets[idx]
                        code.criar_histograma(dfs[nome], nome, f"{nome}_valor")
                    else:
                        print("Opção inválida.")
                except ValueError:
                    print("Digite um número válido.")

            case '9':
                code.mostrar_distribuicao_normal(dfs)
            case '20':
                print("\nObrigado pela nota professor! :)")
                break

            case '0':
                print("\nEncerrando o programa...")
                break

            case _:
                print("\nOpção inválida. Tente novamente.")

if __name__ == "__main__":
    main()