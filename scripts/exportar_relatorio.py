import sqlite3
import pandas as pd
from pathlib import Path

# Caminhos absolutos seguros
BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "data" / "historico_predicoes.db"
OUTPUT_REPORT_PATH = BASE_DIR / "data" / "relatorio_predicoes_exportado.csv"

def exportar_relatorio():
    print("📊 Conectando ao Banco de Dados SQLite para extração de relatório...")
    
    if not DB_PATH.exists():
        print(f"❌ Banco de dados não encontrado em: {DB_PATH}")
        print("Certifique-se de executar algumas inferências na API antes de gerar o relatório.")
        return

    try:
        # Abre conexão com o SQLite
        conn = sqlite3.connect(DB_PATH)
        
        # Lê a tabela inteira usando Pandas para facilitar a análise
        query = "SELECT id, data_hora, classe_predita, confianca, ood FROM predicoes ORDER BY id DESC"
        df = pd.read_sql(query, conn)
        conn.close()

        if df.empty:
            print("⚠️ O banco de dados está vazio. Nenhuma predição foi registrada ainda.")
            return

        # Estatísticas Resumidas
        total_predicoes = len(df)
        total_ood = df['ood'].sum()
        confianca_media = df['confianca'].mean() * 100

        print("\n" + "="*50)
        print("📈 RESUMO EXECUTIVO DO SISTEMA MLOps")
        print("="*50)
        print(f"Total de Inferências Realizadas: {total_predicoes}")
        print(f"Total de Anomalias/OOD Detectadas: {total_ood}")
        print(f"Confiança Média do Modelo: {confianca_media:.2f}%")
        print("="*50 + "\n")

        # Exporta os dados detalhados para CSV (padrão corporativo/relatório)
        df.to_csv(OUTPUT_REPORT_PATH, index=False, encoding='utf-8-sig')
        print(f"💾 Relatório detalhado exportado com sucesso para: {OUTPUT_REPORT_PATH}")

    except Exception as e:
        print(f"❌ Erro crítico ao exportar o relatório: {e}")

if __name__ == "__main__":
    exportar_relatorio()