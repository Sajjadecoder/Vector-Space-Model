import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from query_processor import load_all_indexes, retrieve, ALPHA


class QueryProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vector Space Model - Query Processor")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        self.setup_styles()
        
        self.load_indexes()
        
        self.build_gui()
        
    def setup_styles(self):
        """Configure ttk styles for a clean appearance."""
        style = ttk.Style()
        style.theme_use('clam')
        
        primary_dark = "#2c3e50"    
        primary_light = "#ecf0f1"     
        accent_blue = "#0066cc"       
        text_dark = "#2c3e50"         
        text_light = "#666666" 
        border_color = "#bdc3c7"      
        
        style.configure('TFrame', background=primary_light)
        style.configure('TLabelframe', background=primary_light, foreground=text_dark)
        style.configure('TLabelframe.Label', background=primary_light, foreground=primary_dark, font=('Segoe UI', 10, 'bold'))
        
        style.configure('TLabel', background=primary_light, foreground=text_dark, font=('Segoe UI', 10))
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), background=primary_light, foreground=primary_dark)
        style.configure('Info.TLabel', font=('Segoe UI', 9), background=primary_light, foreground=text_light)
        
        style.configure('TButton', font=('Segoe UI', 10), background=primary_light)
        style.map('TButton', 
                  background=[('active', accent_blue), ('pressed', '#004499')],
                  foreground=[('active', 'white'), ('pressed', 'white')])
        
        style.configure('Search.TButton', font=('Segoe UI', 11, 'bold'), background=accent_blue, foreground='white')
        style.map('Search.TButton',
                  background=[('active', '#004499'), ('pressed', '#003366')],
                  foreground=[('active', 'white'), ('pressed', 'white')])
        
        # Treeview style
        style.configure('Treeview', 
                       background='white',
                       foreground=text_dark,
                       fieldbackground='white',
                       font=('Segoe UI', 10))
        style.configure('Treeview.Heading',
                       background=primary_dark,
                       foreground='white',
                       font=('Segoe UI', 10, 'bold'),
                       borderwidth=1)
        style.map('Treeview',
                 background=[('selected', accent_blue)],
                 foreground=[('selected', 'white')])
        
        self.root.configure(bg=primary_light)
    
    def load_indexes(self):
        """Load all indexes from disk."""
        try:
            self.tfidf, self.idf, self.inverted_idx, self.doc_names = load_all_indexes()
            self.alpha = ALPHA
            self.indexes_loaded = True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load indexes:\n{str(e)}")
            self.indexes_loaded = False
    
    def build_gui(self):
        """Build the GUI components."""
        if not self.indexes_loaded:
            self.root.withdraw()
            return
        
        # Header Frame
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=15, pady=15)
        
        title_label = ttk.Label(header_frame, text="Query Processor", style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Info Frame
        info_frame = ttk.LabelFrame(self.root, text="Corpus Information", padding=10)
        info_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        corpus_size = len(self.doc_names)
        vocab_size = len(self.idf)
        
        info_text = f"Documents: {corpus_size}  |  Vocabulary Size: {vocab_size}  |  Similarity Threshold (α): {self.alpha}"
        info_label = ttk.Label(info_frame, text=info_text, style='Info.TLabel')
        info_label.pack()
        
        # Search Frame
        search_frame = ttk.LabelFrame(self.root, text="Search Query", padding=10)
        search_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        query_label = ttk.Label(search_frame, text="Enter Query:")
        query_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.query_entry = tk.Text(search_frame, height=3, width=80, font=('Segoe UI', 10), wrap=tk.WORD)
        self.query_entry.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.query_entry.bind('<Control-Return>', lambda e: self.search_query())
        
        button_frame = ttk.Frame(search_frame)
        button_frame.pack(fill=tk.X, pady=(0, 5))
        
        search_btn = ttk.Button(button_frame, text="Search", command=self.search_query, style='Search.TButton')
        search_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear_query)
        clear_btn.pack(side=tk.LEFT)
        
        help_label = ttk.Label(search_frame, text="Tip: Press Ctrl+Enter to search", style='Info.TLabel')
        help_label.pack(anchor=tk.W)
        
        # Results Frame
        results_frame = ttk.LabelFrame(self.root, text="Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        # Create Treeview for results
        tree_frame = ttk.Frame(results_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        hsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        
        columns = ('Rank', 'Score', 'Document ID', 'Filename')
        self.results_tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            height=15,
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set
        )
        
        vsb.config(command=self.results_tree.yview)
        hsb.config(command=self.results_tree.xview)
        
        # Configure columns
        self.results_tree.column('#0', width=0, stretch=tk.NO)
        self.results_tree.column('Rank', anchor=tk.CENTER, width=60)
        self.results_tree.column('Score', anchor=tk.CENTER, width=120)
        self.results_tree.column('Document ID', anchor=tk.CENTER, width=100)
        self.results_tree.column('Filename', anchor=tk.W, width=300)
        
        # Create headings
        self.results_tree.heading('#0', text='', anchor=tk.W)
        self.results_tree.heading('Rank', text='Rank', anchor=tk.CENTER)
        self.results_tree.heading('Score', text='Similarity Score', anchor=tk.CENTER)
        self.results_tree.heading('Document ID', text='Doc ID', anchor=tk.CENTER)
        self.results_tree.heading('Filename', text='Filename', anchor=tk.W)
        
        self.results_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Status Frame
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Ready", style='Info.TLabel')
        self.status_label.pack(anchor=tk.W)
    
    def search_query(self):
        query = self.query_entry.get('1.0', tk.END).strip()
        
        if not query:
            messagebox.showwarning("Warning", "Please enter a query.")
            return
        
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Update status
        self.status_label.config(text="Searching...")
        self.root.update_idletasks()
        
        try:
            # Retrieve results
            results = retrieve(query, self.tfidf, self.idf, self.inverted_idx, self.doc_names, self.alpha)
            
            if not results:
                self.status_label.config(text=f"No results found for: \"{query}\"")
                messagebox.showinfo("No Results", "No documents match your query above the threshold.")
                return
            
            # Populate results
            for rank, (doc_id, filename, score) in enumerate(results, 1):
                self.results_tree.insert(
                    '',
                    tk.END,
                    values=(rank, f"{score:.6f}", doc_id, filename)
                )
            
            self.status_label.config(text=f"Found {len(results)} result(s) for: \"{query}\"")
        
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self.status_label.config(text="Error occurred during search")
    
    def clear_query(self):
        self.query_entry.delete('1.0', tk.END)
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.status_label.config(text="Ready")
        self.query_entry.focus()


def main():
    root = tk.Tk()
    app = QueryProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
