import joblib
import numpy as np
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "mnist_best_model.joblib"
KB_PATH = BASE_DIR / "data" / "knowledge_base.json"
DB_PATH = BASE_DIR / "data" / "historico_predicoes.db" # Banco de dados SQLite

class ModelService:
    def __init__(self):
        self.model = None
        self.vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
        self.kb_texts = []
        self.kb_vectors = None
        self.limiar_ood = 0.60 # Limiar de confiança para Out-of-Distribution

    def _inicializar_banco(self):
        """Cria a tabela de histórico se não existir."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predicoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_hora TEXT,
                classe_predita INTEGER,
                confianca REAL,
                ood BOOLEAN
            )
        ''')
        conn.commit()
        conn.close()

    def _salvar_historico(self, classe: int, confianca: float, ood: bool):
        """Salva a predição para futura extração de relatórios."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predicoes (data_hora, classe_predita, confianca, ood) VALUES (?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), classe, confianca, ood)
        )
        conn.commit()
        conn.close()

    def load_model(self):
        try:
            self.model = joblib.load(MODEL_PATH)
            with open(KB_PATH, 'r', encoding='utf-8') as f:
                self.kb_texts = list(json.load(f).values())
            self.kb_vectors = self.vectorizer.fit_transform(self.kb_texts)
            self._inicializar_banco()
            print("🚀 Modelo, RAG e Banco SQLite (Histórico) inicializados.")
        except Exception as e:
            raise RuntimeError(f"Erro no Lifespan: {e}")

    def recuperar_contexto_rag(self, query: str) -> str:
        query_vec = self.vectorizer.transform([query])
        similaridades = cosine_similarity(query_vec, self.kb_vectors).flatten()
        return self.kb_texts[np.argmax(similaridades)]

    def predict_with_rag(self, features: list[float]) -> dict:
        if self.model is None:
            raise RuntimeError("Serviços offline.")
        
        input_array = np.array(features).reshape(1, -1)
        prediction = int(self.model.predict(input_array)[0])
        confidence = float(np.max(self.model.predict_proba(input_array)[0]))
        
        # Implementação OOD (Out-of-Distribution)
        is_ood = confidence < self.limiar_ood
        if is_ood:
            prediction = -1 # Código para anomalia/desconhecido
            contexto = "ALERTA OOD: O traço não se assemelha a nenhum dígito treinado com confiança suficiente."
        else:
            contexto = self.recuperar_contexto_rag(f"O número {prediction} é")
        
        # Salva no histórico SQLite
        self._salvar_historico(prediction, confidence, is_ood)
        
        return {
            "classe_predita": prediction,
            "confianca": confidence,
            "contexto_extra": contexto
        }

ml_service = ModelService()