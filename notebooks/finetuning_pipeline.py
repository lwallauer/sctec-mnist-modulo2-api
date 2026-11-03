import joblib
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Caminho absoluto seguro para a pasta de modelos
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "mnist_best_model.joblib"

def aplicar_data_augmentation(X):
    """Injeta ruído gaussiano para simular traços trêmulos e imperfeitos."""
    ruido = np.random.normal(loc=0.0, scale=0.15, size=X.shape)
    X_ruidoso = X + ruido
    return np.clip(X_ruidoso, 0., 1.)

def evoluir_para_rede_neural():
    print("🚀 Iniciando Pipeline de Treinamento e Evolução (MLOps - Fase 1)...")
    print("💡 Substituindo o KNN antigo por uma Rede Neural (MLP) atualizável.")

    print("📥 Baixando dados reais do MNIST (isso pode levar um minuto)...")
    mnist = fetch_openml('mnist_784', version=1, parser='liac-arff', as_frame=False)
    X_real = mnist.data.astype('float32') / 255.0
    y_real = mnist.target.astype('int')
    
    # Seleciona 10.000 amostras para treinar um modelo robusto rapidamente
    print("🎯 Selecionando lote de 10.000 imagens e aplicando Data Augmentation...")
    indices = np.random.choice(len(X_real), size=10000, replace=False)
    X_treino = X_real[indices]
    y_treino = y_real[indices]

    X_treino_ruidoso = aplicar_data_augmentation(X_treino)

    print("⚙️ Treinando a nova Rede Neural (MLPClassifier)...")
    # Configuração da Rede Neural com suporte nativo a fine-tuning futuro (warm_start=True)
    modelo_mlp = MLPClassifier(
        hidden_layer_sizes=(128,), 
        max_iter=25, 
        solver='adam', 
        verbose=True, 
        random_state=42,
        warm_start=True
    )
    
    modelo_mlp.fit(X_treino_ruidoso, y_treino)
    
    print("\n📊 Avaliando o desempenho do novo modelo:")
    predicoes = modelo_mlp.predict(X_treino_ruidoso)
    acuracia = accuracy_score(y_treino, predicoes)
    print(f"Acurácia no lote de treinamento com ruído: {acuracia * 100:.2f}%")
    
    # Sobrescreve o artefato com a versão evoluída
    joblib.dump(modelo_mlp, MODEL_PATH)
    print(f"💾 Nova Rede Neural serializada com sucesso em: {MODEL_PATH}")

if __name__ == "__main__":
    evoluir_para_rede_neural()