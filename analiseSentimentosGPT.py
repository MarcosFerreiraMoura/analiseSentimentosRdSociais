import pandas as pd
import google.generativeai as genai
import os
import time
import re # Importado para limpeza de HTML
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# --- 1. Configuração da API do Gemini ---
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("Erro: A variável de ambiente GOOGLE_API_KEY não foi definida.")
    exit()

# Configurações do modelo
generation_config = {
  "temperature": 0.1,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 10, 
}

safety_settings = [
  {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Inicializa o modelo
model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# --- 2. Funções ---

def carregar_dados():
    """Carrega os dados e padroniza os rótulos de sentimento."""
    try:
        data = pd.read_csv(r'data\imdb-reviews-pt-br.csv')
       
        # MUDANÇA 1: Padronizar os rótulos para que sejam iguais aos do Gemini
        # Mapeia 'pos' para 'positivo' e 'neg' para 'negativo'
        label_map = {'pos': 'positivo', 'neg': 'negativo'}
        data['sentiment'] = data['sentiment'].map(label_map)
        
        return data
    except FileNotFoundError:
        print("Erro: Arquivo 'data\\imdb-reviews-pt-br.csv' não encontrado.")
        return None

def clean_html(text):
    """Remove tags HTML do texto."""
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', text)

def analisar_sentimento_gemini(texto):
    """Envia o texto para a API Gemini e retorna o sentimento."""
    # MUDANÇA 2: Melhorar o prompt para ser mais robusto
    prompt = f"""
        Analisa o sentimento de um texto usando a API do Gemini.
        Retorna 'positivo' ou 'negativo'.
    
    Avaliação: "{texto}"

    Sentimento:
    """
    try:
        response = model.generate_content(prompt)
        resultado = response.text.strip().lower()
        
        if resultado in ['positivo', 'negativo']:
            return resultado
        else:
            # MUDANÇA 3: Imprimir respostas inesperadas para depuração
            print(f"\n[AVISO] Resposta inesperada do modelo: '{resultado}'")
            return "nao_classificado"

    except Exception as e:
        print(f"\n[ERRO] Ocorreu um erro ao chamar a API: {e}")
        return "erro"

# --- 3. Execução Principal ---
if __name__ == "__main__":
    data = carregar_dados()

    if data is not None:
        # Usando as mesmas 100 avaliações para consistência.
        # Mude o valor de 'n' se desejar.
        amostra_df = data.sample(n=10, random_state=42)
        
        print(f"Iniciando análise de sentimentos em {len(amostra_df)} avaliações usando Gemini...")

        predicoes = []
        for texto in tqdm(amostra_df['text_pt'], desc="Analisando avaliações"):
            # MUDANÇA 4: Limpar o texto ANTES de enviar para a API
            texto_limpo = clean_html(texto)
            predicao = analisar_sentimento_gemini(texto_limpo)
            predicoes.append(predicao)
            time.sleep(1.5) 

        amostra_df['predicao_gemini'] = predicoes
        amostra_analisada = amostra_df[amostra_df['predicao_gemini'].isin(['positivo', 'negativo'])]

        y_real = amostra_analisada['sentiment']
        y_pred = amostra_analisada['predicao_gemini']
        
        # --- 4. Resultados ---
        if not y_real.empty:
            acuracia = accuracy_score(y_real, y_pred)
            print("\n--- Resultados da Análise com Gemini ---")
            print(f"Acurácia do modelo Gemini na amostra: {acuracia:.2%}")
            
            print("\nRelatório de Classificação:")
            print(classification_report(y_real, y_pred))

            print("\nExemplos de classificação (reais vs. previstos):")
            print(amostra_analisada[['sentiment', 'predicao_gemini']].head(10))
        else:
            print("\nNão foi possível calcular a acurácia. Nenhuma avaliação foi analisada com sucesso.")
            # Mostra o que foi retornado para ajudar a depurar
            print("Verifique as respostas recebidas do modelo:")
            print(amostra_df[['sentiment', 'predicao_gemini']].head(10))