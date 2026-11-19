
from pathlib import Path

from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "https://acessobairro.vercel.app"
        ]
    }
})

# Caminho da pasta onde este arquivo está localizado
PASTA_API = Path(__file__).resolve().parent

# Carregar os modelos treinados
modelo_barreiras = joblib.load(PASTA_API / "modelo_barreiras.joblib")
modelo_prioridade = joblib.load(PASTA_API / "modelo_prioridade.joblib")

# Valores aceitos pelos modelos
CATEGORIAS = [
    "CALCADA_DANIFICADA",
    "RAMPA_AUSENTE",
    "PASSAGEM_OBSTRUIDA",
    "RAMPA_INADEQUADA",
    "TRAVESSIA_INADEQUADA"
]

IMPACTOS = ["BAIXO", "MEDIO", "ALTO"]

ROTAS_ALTERNATIVAS = ["SIM", "NAO"]


# Rota inicial para verificar se a API está funcionando
@app.route("/", methods=["GET"])
def inicio():
    return jsonify({
        "mensagem": "API de Inteligência Artificial - AcessoBairro",
        "status": "online"
    })


# Modelo 1: Classificação de barreiras
@app.route("/api/classificar-barreira", methods=["POST"])
def classificar_barreira():
    dados = request.get_json(silent=True)

    if not isinstance(dados, dict):
        return jsonify({
            "erro": "Envie os dados em formato JSON."
        }), 400

    descricao = dados.get("descricao")

    if not isinstance(descricao, str) or not descricao.strip():
        return jsonify({
            "erro": "Informe uma descricao valida."
        }), 400

    descricao = descricao.strip()

    previsao = modelo_barreiras.predict([descricao])

    return jsonify({
        "descricao": descricao,
        "categoria_sugerida": str(previsao[0])
    })


# Modelo 2: Classificação de prioridade
@app.route("/api/classificar-prioridade", methods=["POST"])
def classificar_prioridade():
    dados = request.get_json(silent=True)

    if not isinstance(dados, dict):
        return jsonify({
            "erro": "Envie os dados em formato JSON."
        }), 400

    categoria = dados.get("categoria")
    impacto = dados.get("impacto")
    rota_alternativa = dados.get("rota_alternativa")

    # Validar os tipos dos campos
    if not all(
        isinstance(valor, str)
        for valor in [categoria, impacto, rota_alternativa]
    ):
        return jsonify({
            "erro": "Os campos categoria, impacto e rota_alternativa devem ser textos."
        }), 400

    # Padronizar os valores recebidos
    categoria = categoria.strip().upper()
    impacto = impacto.strip().upper()
    rota_alternativa = rota_alternativa.strip().upper()

    # Validar os valores permitidos
    if categoria not in CATEGORIAS:
        return jsonify({
            "erro": "Categoria invalida."
        }), 400

    if impacto not in IMPACTOS:
        return jsonify({
            "erro": "Impacto invalido. Utilize BAIXO, MEDIO ou ALTO."
        }), 400

    if rota_alternativa not in ROTAS_ALTERNATIVAS:
        return jsonify({
            "erro": "Rota alternativa invalida. Utilize SIM ou NAO."
        }), 400

    entrada = pd.DataFrame([{
        "categoria": categoria,
        "impacto": impacto,
        "rota_alternativa": rota_alternativa
    }])

    previsao = modelo_prioridade.predict(entrada)

    return jsonify({
        "categoria": categoria,
        "impacto": impacto,
        "rota_alternativa": rota_alternativa,
        "prioridade_sugerida": str(previsao[0])
    })

# Classificação completa: barreira + prioridade
@app.route("/api/analisar-relato", methods=["POST"])
def analisar_relato():
    dados = request.get_json(silent=True)

    if not isinstance(dados, dict):
        return jsonify({
            "erro": "Envie os dados em formato JSON."
        }), 400

    descricao = dados.get("descricao")
    impacto = dados.get("impacto")
    rota_alternativa = dados.get("rota_alternativa")

    if not isinstance(descricao, str) or not descricao.strip():
        return jsonify({
            "erro": "Informe uma descricao válida."
        }), 400

    if not isinstance(impacto, str) or not isinstance(rota_alternativa, str):
        return jsonify({
            "erro": "Informe impacto e rota_alternativa como textos."
        }), 400

    impacto = impacto.strip().upper()
    rota_alternativa = rota_alternativa.strip().upper()

    if impacto not in IMPACTOS:
        return jsonify({
            "erro": "Impacto inválido. Utilize BAIXO, MEDIO ou ALTO."
        }), 400

    if rota_alternativa not in ROTAS_ALTERNATIVAS:
        return jsonify({
            "erro": "Rota alternativa inválida. Utilize SIM ou NAO."
        }), 400

    # Primeiro modelo: identifica a categoria da barreira
    categoria = str(
        modelo_barreiras.predict([descricao.strip()])[0]
    )

    # Segundo modelo: estima a prioridade usando a categoria prevista
    entrada = pd.DataFrame([{
        "categoria": categoria,
        "impacto": impacto,
        "rota_alternativa": rota_alternativa
    }])

    prioridade = str(modelo_prioridade.predict(entrada)[0])

    return jsonify({
        "descricao": descricao.strip(),
        "categoria_sugerida": categoria,
        "impacto": impacto,
        "rota_alternativa": rota_alternativa,
        "prioridade_sugerida": prioridade
    })

if __name__ == "__main__":
    app.run(debug=True)
