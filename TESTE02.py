"""
App unificado de Análise de Sentimentos no Reddit usando as APIs do Reddit (PRAW)
e do Gemini (google-generativeai).

• Busca posts e comentários no Reddit por palavra-chave.
• Classifica cada texto via Gemini como "positivo", "negativo" ou "neutro".
• (Opcional) Calcula rótulo VADER em paralelo para comparações rápidas.
• Exibe um gráfico de barras e uma tabela filtrável em Tkinter.

REQUISITOS
----------
pip install praw google-generativeai vaderSentiment pandas matplotlib

VARIÁVEIS DE AMBIENTE (obrigatório)
------------------------------------
REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
GOOGLE_API_KEY
"""

import os
import time
import re
import threading
import queue  # CORREÇÃO: Importar a biblioteca de fila
from dataclasses import dataclass

import pandas as pd
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import google.generativeai as genai

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =====================
# ===== CONFIG ========
# =====================
# CORREÇÃO: Remover chaves padrão para forçar o uso de variáveis de ambiente.
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

SUBREDDIT = "all"
POST_LIMIT = 8
COMMENT_LIMIT = 5

GEN_MODEL_NAME = "gemini-1.5-flash"
GEN_TEMPERATURE = 0.1
GEN_TOP_P = 1
GEN_TOP_K = 1
GEN_MAX_TOKENS = 10
GEN_SLEEP_BETWEEN_CALLS = 1.1

ENABLE_VADER_COLUMN = True


# =====================
# ===== UTILS =========
# =====================
HTML_TAGS = re.compile(r"<.*?>")
MULTISPACE = re.compile(r"\s+")
URL_PATTERN = re.compile(r"https?://\S+")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = re.sub(HTML_TAGS, " ", text)
    t = re.sub(URL_PATTERN, " ", t)
    t = t.replace("\n", " ")
    t = re.sub(MULTISPACE, " ", t).strip()
    return t

def normalize_label(label: str) -> str:
    if not isinstance(label, str):
        return "nao_classificado"
    l = label.strip().lower()
    if "positivo" in l:
        return "positivo"
    if "negativo" in l:
        return "negativo"
    if "neutro" in l or "neutra" in l or "neither" in l:
        return "neutro"
    return "nao_classificado"


@dataclass
class Config:
    subreddit: str = SUBREDDIT
    post_limit: int = POST_LIMIT
    comment_limit: int = COMMENT_LIMIT


# =========================
# ===== REDDIT (PRAW) =====
# =========================
def make_reddit_client():
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        raise ValueError("Configure as variáveis de ambiente do Reddit: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT")
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

def collect_reddit_texts(keyword: str, cfg: Config) -> list[dict]:
    reddit = make_reddit_client()
    results = []
    posts = reddit.subreddit(cfg.subreddit).search(keyword, limit=cfg.post_limit)
    for post in posts:
        if post.title:
            results.append({"origem": "titulo", "texto": clean_text(post.title)})
        try:
            post.comments.replace_more(limit=0)
            for c in post.comments.list()[: cfg.comment_limit]:
                results.append({"origem": "comentario", "texto": clean_text(getattr(c, "body", ""))})
        except Exception:
            continue
    return [r for r in results if r["texto"]]


def load_manual_texts(filename: str) -> list[dict]:
    """
    Lê um CSV já existente com a coluna 'comentario/review'
    e converte para o mesmo formato usado no pipeline normal.
    """
    import csv
    results = []
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            texto = clean_text(row.get("comentario/review", ""))
            if texto:
                results.append({"origem": "manual", "texto": texto})
    return results



import csv

# Função para garantir que a pasta exista
def garantir_pasta(pasta):
    if not os.path.exists(pasta):
        os.makedirs(pasta)

# Atualizando a função save_reddit_texts
def save_reddit_texts(texts: list[dict], filename: str = "reddit_data.csv"):
    """
    Salva os textos coletados do Reddit em um CSV na pasta 'dadosPesquisa'.
    """
    if not texts:
        print("[AVISO] Nenhum texto para salvar.")
        return
    
    garantir_pasta("dadosPesquisa")
    caminho_arquivo = os.path.join("dadosPesquisa", filename)
    
    with open(caminho_arquivo, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["id", "comentario/review"])  # cabeçalho
        for idx, item in enumerate(texts, start=1):
            writer.writerow([idx, item.get("texto", "")])
    
    print(f"[INFO] Dados crus salvos em: {caminho_arquivo}")

# Atualizando a função save_results
def save_results(df: pd.DataFrame, filename: str = "reddit_results.csv"):
    """
    Salva os resultados da classificação em um CSV na pasta 'dadosAnalisados'.
    """
    if df.empty:
        print("[AVISO] Nenhum resultado para salvar.")
        return

    garantir_pasta("dadosAnalisados")
    caminho_arquivo = os.path.join("dadosAnalisados", filename)

    import csv
    cols = ["id", "origem", "comentario/review", "sent_gemini"]
    if "sent_vader" in df.columns:
        cols.append("sent_vader")

    with open(caminho_arquivo, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(cols)  # cabeçalho
        
        for idx, row in enumerate(df.itertuples(index=False), start=1):
            line = [idx, getattr(row, "origem"), getattr(row, "texto"), getattr(row, "sent_gemini")]
            if "sent_vader" in df.columns:
                line.append(getattr(row, "sent_vader"))
            writer.writerow(line)

    print(f"[INFO] Resultados classificados salvos em: {caminho_arquivo}")







# ==========================
# ===== GEMINI (LLM) =======
# ==========================
def make_gemini_model():
    if not GOOGLE_API_KEY:
        raise ValueError("Configure a variável de ambiente GOOGLE_API_KEY.")
    genai.configure(api_key=GOOGLE_API_KEY)
    generation_config = {
        "temperature": GEN_TEMPERATURE, "top_p": GEN_TOP_P,
        "top_k": GEN_TOP_K, "max_output_tokens": GEN_MAX_TOKENS,
    }
    return genai.GenerativeModel(
        model_name=GEN_MODEL_NAME, generation_config=generation_config
    )

GEM_PROMPT_TEMPLATE = """
Classifique o sentimento do texto a seguir. Responda com UMA PALAVRA:
"positivo", "negativo" ou "neutro". Não explique.

Texto: "{texto}"

Sentimento:
"""

def classify_with_gemini(model, texto: str) -> str:
    prompt = GEM_PROMPT_TEMPLATE.format(texto=texto)
    try:
        resp = model.generate_content(prompt)
        return normalize_label(getattr(resp, "text", ""))
    except Exception as e:
        print(f"[AVISO] Erro ao chamar Gemini: {e}")
        return "nao_classificado"

_vader = SentimentIntensityAnalyzer()
def classify_with_vader(texto: str) -> str:
    s = _vader.polarity_scores(texto)
    if s["compound"] >= 0.05: return "positivo"
    elif s["compound"] <= -0.05: return "negativo"
    else: return "neutro"


# =====================================
# ===== PIPELINE + UI (Tkinter) =======
# =====================================
class App:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Análise de Sentimento Reddit (Gemini)")
        master.geometry("1000x650")

        # --- Widgets ---
        top = tk.Frame(master); top.pack(pady=8)
        tk.Label(top, text="Palavra-chave:").pack(side=tk.LEFT)
        self.entry = tk.Entry(top, width=40); self.entry.pack(side=tk.LEFT, padx=6)
        self.search_button = tk.Button(top, text="Pesquisar", command=self.on_search); self.search_button.pack(side=tk.LEFT)

        self.file_button = tk.Button(top, text="Adicionar Arquivos", command=self.on_add_file)
        self.file_button.pack(side=tk.LEFT, padx=6)

        filt = tk.Frame(master); filt.pack(pady=5)
        self.filtro_var = tk.StringVar(value="Todos")
        ttk.Combobox(
            filt, textvariable=self.filtro_var,
            values=["Todos", "positivo", "negativo", "neutro", "nao_classificado"],
            state="readonly", width=18,
        ).pack(side=tk.LEFT)
        tk.Button(filt, text="Filtrar", command=self.apply_filter).pack(side=tk.LEFT, padx=6)

        self.fig, self.ax = plt.subplots(figsize=(6, 3.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(pady=4)

        table_fr = tk.Frame(master); table_fr.pack(fill=tk.BOTH, expand=True, pady=6)
        self.columns = ("origem", "sent_gemini", "sent_vader", "texto") if ENABLE_VADER_COLUMN else ("origem", "sent_gemini", "texto")
        self.tree = ttk.Treeview(table_fr, columns=self.columns, show="headings")
        for col in self.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=160 if col != "texto" else 600, anchor=tk.W)
        self.tree.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(table_fr, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.status = tk.StringVar(value="Pronto."); tk.Label(master, textvariable=self.status, anchor=tk.W).pack(fill=tk.X)

        # --- State ---
        self.df_total = pd.DataFrame()
        self.model = None
        self.thread_queue = queue.Queue() # CORREÇÃO: Fila para comunicação entre threads
        self.master.after(100, self.process_queue) # CORREÇÃO: Iniciar o processador da fila

    def set_status(self, msg: str):
        self.status.set(msg)

    def on_search(self):
        kw = self.entry.get().strip()
        if not kw:
            messagebox.showwarning("Aviso", "Digite uma palavra-chave.")
            return
        
        self.search_button.config(state="disabled") # Desabilitar botão para evitar buscas múltiplas
        self.set_status("Iniciando busca...")
        # CORREÇÃO: Limpar resultados antigos imediatamente
        self.df_total = pd.DataFrame()
        self.update_chart()
        self.fill_table(self.df_total)
        
        threading.Thread(target=self._run_pipeline, args=(kw,), daemon=True).start()

    def on_add_file(self):
        filename = filedialog.askopenfilename(
            title="Selecione um arquivo",
            filetypes=[("CSV Files", "*.csv"), ("Todos os arquivos", "*.*")]
        )
        if not filename:
            return  # usuário cancelou

        self.search_button.config(state="disabled")
        self.file_button.config(state="disabled")
        self.set_status(f"Lendo arquivo: {filename}")

        # limpa resultados anteriores
        self.df_total = pd.DataFrame()
        self.update_chart()
        self.fill_table(self.df_total)

        # processa em thread separada
        threading.Thread(target=self._run_pipeline_file, args=(filename,), daemon=True).start()




    # CORREÇÃO: Este método agora roda em uma thread separada e se comunica via fila
    def _run_pipeline(self, kw: str):
        try:
            self.thread_queue.put(("status", "Coletando textos do Reddit…"))
            texts = collect_reddit_texts(kw, Config())
            save_reddit_texts(texts, filename=f"reddit_{kw}_raw.csv")

            if not texts:
                self.thread_queue.put(("message", ("info", "Nenhum texto encontrado para o termo.")))
                self.thread_queue.put(("done", None))
                return
        except Exception as e:
            self.thread_queue.put(("message", ("error", f"Erro ao buscar no Reddit:\n{e}")))
            self.thread_queue.put(("done", None))
            return
           
        try:
            if self.model is None:
                self.thread_queue.put(("status", "Inicializando modelo Gemini…"))
                self.model = make_gemini_model()
        except Exception as e:
            self.thread_queue.put(("message", ("error", f"Erro ao inicializar o Gemini:\n{e}")))
            self.thread_queue.put(("done", None))
            return

        rows = []
        for i, item in enumerate(texts, start=1):
            txt = item["texto"]
            self.thread_queue.put(("status", f"Classificando com Gemini… {i}/{len(texts)}"))
            g_label = classify_with_gemini(self.model, txt)
            v_label = classify_with_vader(txt) if ENABLE_VADER_COLUMN else None
            row_data = {"origem": item["origem"], "texto": txt, "sent_gemini": g_label}
            if ENABLE_VADER_COLUMN:
                row_data["sent_vader"] = v_label
            rows.append(row_data)
            time.sleep(GEN_SLEEP_BETWEEN_CALLS)
        
        df_results = pd.DataFrame(rows)
        save_results(df_results, filename=f"reddit_{kw}_results.csv")
        self.thread_queue.put(("results", df_results))
        self.thread_queue.put(("done", None))

    # CORREÇÃO: Novo método para processar a fila na thread principal
    def process_queue(self):
        try:
            msg_type, data = self.thread_queue.get_nowait()
            
            if msg_type == "status":
                self.set_status(data)
            elif msg_type == "results":
                self.df_total = data
                self.update_chart()
                self.fill_table(self.df_total)
            elif msg_type == "message":
                msg_level, msg_text = data
                if msg_level == "info": messagebox.showinfo("Informação", msg_text)
                elif msg_level == "error": messagebox.showerror("Erro", msg_text)
            elif msg_type == "done":
                self.set_status("Concluído.")
                self.search_button.config(state="normal") # Reabilitar o botão
                self.file_button.config(state="normal")

        
        except queue.Empty:
            pass # Fila vazia, não faz nada
        finally:
            self.master.after(100, self.process_queue) # Reagendar a verificação

    def update_chart(self):
        self.ax.clear()
        
        if not self.df_total.empty:
            # --- Gemini ---
            counts_gemini = self.df_total["sent_gemini"].value_counts()
            colors_g = {'positivo': 'green', 'negativo': 'red', 'neutro': 'gray', 'nao_classificado': 'orange'}
            bar_colors_g = [colors_g.get(label, 'blue') for label in counts_gemini.index]
            
            # --- VADER ---
            if ENABLE_VADER_COLUMN and "sent_vader" in self.df_total.columns:
                counts_vader = self.df_total["sent_vader"].value_counts()
                colors_v = {'positivo': 'green', 'negativo': 'red', 'neutro': 'gray', 'nao_classificado': 'orange'}
                bar_colors_v = [colors_v.get(label, 'blue') for label in counts_vader.index]
            else:
                counts_vader, bar_colors_v = pd.Series(), []

            # --- Subplots lado a lado ---
            self.fig.clf()
            ax1 = self.fig.add_subplot(1,2,1)
            ax2 = self.fig.add_subplot(1,2,2)

            ax1.bar(counts_gemini.index, counts_gemini.values, color=bar_colors_g)
            ax1.set_title("Distribuição Gemini")
            ax1.set_ylabel("Quantidade")

            if not counts_vader.empty:
                ax2.bar(counts_vader.index, counts_vader.values, color=bar_colors_v)
                ax2.set_title("Distribuição VADER")
                ax2.set_ylabel("Quantidade")

            self.fig.tight_layout()

        self.canvas.draw()


    def fill_table(self, df: pd.DataFrame):
        for row in self.tree.get_children():
            self.tree.delete(row)
        
        # MELHORIA: Lógica simplificada sem duplicar o laço
        for _, r in df.iterrows():
            texto_truncado = (r["texto"][:180] + "…") if len(r["texto"]) > 180 else r["texto"]
            values = [r["origem"], r["sent_gemini"]]
            if ENABLE_VADER_COLUMN:
                values.append(r["sent_vader"])
            values.append(texto_truncado)
            self.tree.insert("", tk.END, values=tuple(values))

    def apply_filter(self):
        f = self.filtro_var.get()
        if f == "Todos":
            self.fill_table(self.df_total)
        else:
            if not self.df_total.empty:
                filtered_df = self.df_total[self.df_total["sent_gemini"] == f]
                self.fill_table(filtered_df)
            else:
                self.fill_table(pd.DataFrame())

    def _run_pipeline_file(self, filename: str):
        try:
            self.thread_queue.put(("status", "Carregando comentários do arquivo…"))
            texts = load_manual_texts(filename)

            if not texts:
                self.thread_queue.put(("message", ("info", "Nenhum texto encontrado no arquivo.")))
                self.thread_queue.put(("done", None))
                return
        except Exception as e:
            self.thread_queue.put(("message", ("error", f"Erro ao ler o arquivo:\n{e}")))
            self.thread_queue.put(("done", None))
            return

        try:
            if self.model is None:
                self.thread_queue.put(("status", "Inicializando modelo Gemini…"))
                self.model = make_gemini_model()
        except Exception as e:
            self.thread_queue.put(("message", ("error", f"Erro ao inicializar o Gemini:\n{e}")))
            self.thread_queue.put(("done", None))
            return

        rows = []
        for i, item in enumerate(texts, start=1):
            txt = item["texto"]
            self.thread_queue.put(("status", f"Classificando com Gemini… {i}/{len(texts)}"))
            g_label = classify_with_gemini(self.model, txt)
            v_label = classify_with_vader(txt) if ENABLE_VADER_COLUMN else None
            row_data = {"origem": item["origem"], "texto": txt, "sent_gemini": g_label}
            if ENABLE_VADER_COLUMN:
                row_data["sent_vader"] = v_label
            rows.append(row_data)
            time.sleep(GEN_SLEEP_BETWEEN_CALLS)

        df_results = pd.DataFrame(rows)
        save_results(df_results, filename=os.path.basename(filename).replace(".csv", "_results.csv"))
        self.thread_queue.put(("results", df_results))
        self.thread_queue.put(("done", None))


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
    