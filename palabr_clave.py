import os
import re
import math
from collections import defaultdict
#from flask import Flask, request, render_template, redirect, url_for
#from werkzeug.utils import secure_filename

#app = Flask(__name__)
#app.static_folder = 'static'
#app.config['UPLOAD_FOLDER'] = 'uploads'

STOP_WORDS = set([
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o",
    "de", "del", "en", "a", "que", "es", "con", "para", "por", "al", "era", "entre",
    "lo", "como", "más", "pero", "sus", "le", "se", "si", "son", "su",
    "ya", "este", "ha", "me", "mi", "sí", "yo", "tu", "te", "su", "quien", "carlos",
    "nos", "nosotros", "vosotros", "vos", "tu", "ellos", "ellas", "tambien", "p", "ino", "desde",
    "nuestro", "nuestra", "vuestro", "vuestra", "su", "cuando", "donde",
    "cual", "cuales", "que", "aquel", "aquella", "aquello", "esos", "dia",
    "esas", "este", "esta", "estos", "estas", "mi", "mis", "tu", "tus", "viene",
    "su", "sus", "nuestro", "nuestra", "nuestros", "nuestras", "vuestro", "sin", "hay",
    "vuestra", "vuestros", "vuestras", "ese", "esa", "esos", "esas", "estoy", "c",
    "the", "and", "a", "an", "of", "to", "in", "is", "that", "it", "mucho",
    "on", "for", "with", "as", "by", "at", "from", "about", "into", "muy", "https", "doi",
    "over", "after", "beneath", "under", "above", "below", "more", "e", 
    "most", "some", "such", "no", "nor", "not", "only", "own", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "uso",
    "should", "now"
])

class AVLNode:
    def __init__(self, key, height=1, left=None, right=None):
        self.key = key
        self.height = height
        self.left = left
        self.right = right

class AVLTree:
    def __init__(self):
        self.root = None

    def insert(self, root, key):
        if not root:
            return AVLNode(key)
        elif key < root.key:
            root.left = self.insert(root.left, key)
        else:
            root.right = self.insert(root.right, key)

        root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))

        balance = self.get_balance(root)

        if balance > 1 and key < root.left.key:
            return self.right_rotate(root)
        if balance < -1 and key > root.right.key:
            return self.left_rotate(root)
        if balance > 1 and key > root.left.key:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)
        if balance < -1 and key < root.right.key:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)

        return root

    def get_height(self, root):
        if not root:
            return 0
        return root.height

    def get_balance(self, root):
        if not root:
            return 0
        return self.get_height(root.left) - self.get_height(root.right)

    def left_rotate(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        return y

    def right_rotate(self, z):
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        return y

    def pre_order(self, root):
        res = []
        if root:
            res.append(root.key)
            res = res + self.pre_order(root.left)
            res = res + self.pre_order(root.right)
        return res

class KeywordExtractor:
    def __init__(self):
        self.avl_tree = AVLTree()

    def tokenize(self, text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [token for token in tokens if token not in STOP_WORDS and not token.isdigit()]

    def calculate_tf(self, tokens):
        tf = defaultdict(float)
        token_count = len(tokens)
        for token in tokens:
            tf[token] += 1.0 / token_count
        return tf

    def calculate_idf(self, corpus):
        idf = defaultdict(float)
        doc_count = len(corpus)
        for document in corpus:
            tokens = set(self.tokenize(document))
            for token in tokens:
                idf[token] += 1.0
        for token in idf:
            idf[token] = math.log(doc_count / (1.0 + idf[token]))
        return idf

    def calculate_tfidf(self, tf, idf):
        tfidf = {term: tf[term] * idf.get(term, 0.0) for term in tf}
        return tfidf

    def extract_keywords(self, document, corpus, n):
        tokens = self.tokenize(document)
        tf = self.calculate_tf(tokens)
        idf = self.calculate_idf(corpus)
        tfidf = self.calculate_tfidf(tf, idf)
        return self.get_top_n_keywords(tfidf, n)

    def get_top_n_keywords(self, tfidf, n):
        sorted_tfidf = sorted(tfidf.items(), key=lambda item: item[1], reverse=True)
        top_keywords = [term for term, score in sorted_tfidf[:n]]
        for keyword in top_keywords:
            self.avl_tree.root = self.avl_tree.insert(self.avl_tree.root, keyword)
        return top_keywords

    def extract_year(self, text):
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
        print("Años encontrados:", years)  # Mostrar todos los años encontrados
        if years:
            return max(years)  # Retornar el año más reciente encontrado
        return None

    def extract_sender_receiver(self, text):
        sender_pattern = r'\b(?:Sincerely|De|Atentamente|Saludos|Cordialmente),?\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)'
        receiver_pattern = r'(?:Señor|Sr\.|Para|Doctor|Dr\.|Licenciado|Estimado|Lic\.|Ingeniero|Ing\.)\s*:?\s*([A-Z][a-zñáéíóú]+(?:\s[A-Z][a-zñáéíóú]+)*)'
        sender = re.findall(sender_pattern, text)
        receiver = re.findall(receiver_pattern, text)
        print("Remitente encontrado:", sender)  # Mostrar remitente encontrado
        print("Destinatario encontrado:", receiver)  # Mostrar destinatario encontrado

        return sender, receiver

def palabr_c(ruta,fn,num_keywords):
    filepath=os.path.join(ruta, fn+"1.txt")
#@app.route('/', methods=['GET', 'POST'])
#def upload_file():
#    if request.method == 'POST':
#        file = request.files['file']

#        num_keywords = int(request.form['num_keywords'])
#        if file:
#            filename = secure_filename(file.filename)
#            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#            file.save(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        document = f.read()

    with open(filepath, 'r', encoding='utf-8') as f:
        corpus = f.readlines()

    extractor = KeywordExtractor()
    keywords = extractor.extract_keywords(document, corpus, num_keywords)
    avl_keywords = extractor.avl_tree.pre_order(extractor.avl_tree.root)
            
    year = extractor.extract_year(document)
    sender, receiver = extractor.extract_sender_receiver(document)

    return keywords, avl_keywords, year, sender, receiver
#    return render_template('index1.html', keywords=keywords, avl_keywords=avl_keywords, year=year, sender=sender, receiver=receiver)

#    return render_template('index1.html')

#if __name__ == '__main__':
#    if not os.path.exists(app.config['UPLOAD_FOLDER']):
#        os.makedirs(app.config['UPLOAD_FOLDER'])
#    app.run(debug=True)