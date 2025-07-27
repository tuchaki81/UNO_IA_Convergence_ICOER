
import json
import numpy as np

def load_icoers(path='ia_responses.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calcular_convergencia(vetores):
    matriz = np.array([[v['SLECMA']] for v in vetores])
    media = np.mean(matriz, axis=0)
    desvios = np.std(matriz, axis=0)
    icu_score = 1 - np.mean(desvios)
    return round(icu_score, 4), media.tolist()

def compute_icoer():
    dados = load_icoers()
    score, media_slecma = calcular_convergencia(dados)
    return {
        "Indice_Convergencia_UNO": score,
        "Media_SLECMA": media_slecma
    }

def main():
    dados = load_icoers()
    score, media_slecma = calcular_convergencia(dados)
    resultado = {
        "Indice_Convergencia_UNO": score,
        "Media_SLECMA": media_slecma
    }
    with open("index_convergente.json", "w") as f:
        json.dump(resultado, f, indent=4)
    print("🧠 Índice de Convergência UNO:", score)

if __name__ == "__main__":
    main()
