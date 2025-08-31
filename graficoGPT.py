import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Distribuição de Sentimentos - CSV")

        # Botão para carregar arquivo
        self.btn_carregar = tk.Button(root, text="Carregar CSV", command=self.carregar_csv)
        self.btn_carregar.pack(pady=10)

        # Figura do Matplotlib
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Frame para tabela
        self.frame_tabela = tk.Frame(root)
        self.frame_tabela.pack(fill=tk.BOTH, expand=True)

        # Treeview (tabela)
        self.tree = ttk.Treeview(self.frame_tabela, columns=("origem", "sent_gpt5", "texto"), show="headings")
        self.tree.heading("origem", text="Origem")
        self.tree.heading("sent_gpt5", text="Sentimento GPT-5")
        self.tree.heading("texto", text="Texto")

        self.tree.column("origem", width=100)
        self.tree.column("sent_gpt5", width=120)
        self.tree.column("texto", width=600)

        self.tree.pack(fill=tk.BOTH, expand=True)

    def carregar_csv(self):
        caminho_csv = filedialog.askopenfilename(
            title="Selecione o arquivo CSV",
            filetypes=[("Arquivos CSV", "*.csv")]
        )
        if not caminho_csv:
            return  # usuário cancelou

        self.update_chart_csv(caminho_csv)

    def update_chart_csv(self, caminho_csv):
        try:
            df = pd.read_csv(caminho_csv, sep=";")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao ler CSV:\n{e}")
            return

        if "sent_gpt5" not in df.columns or "origem" not in df.columns or "texto" not in df.columns:
            messagebox.showerror("Erro", "O CSV precisa ter as colunas: 'origem', 'sent_gpt5' e 'texto'!")
            return

        # Limpa eixos do gráfico
        self.ax.clear()

        # Conta sentimentos
        counts = df["sent_gpt5"].value_counts()

        # Define cores
        colors = {
            'positivo': 'green',
            'negativo': 'red',
            'neutro': 'gray',
            'nao_classificado': 'orange'
        }
        bar_colors = [colors.get(label, 'blue') for label in counts.index]

        # Cria gráfico
        self.ax.bar(counts.index, counts.values, color=bar_colors)
        self.ax.set_title("Distribuição GPT5")
        self.ax.set_ylabel("Quantidade")
        self.fig.tight_layout()

        # Atualiza gráfico no Tkinter
        self.canvas.draw()

        # Atualiza tabela
        for row in self.tree.get_children():
            self.tree.delete(row)

        for _, linha in df.iterrows():
            self.tree.insert("", tk.END, values=(linha["origem"], linha["sent_gpt5"], linha["texto"]))


# Exemplo de uso
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
