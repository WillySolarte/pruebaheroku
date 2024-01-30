
import pickle
#Inicio de nuevas inportaciones
from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
# Importa las dependencias específicas de tu nuevo modelo
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

from clases.vectorizer_transformer import VectorizerTransformer
from clases.qkernel_transformer import QKernelTransformer

#Fin de nuevas inportaciones
def crear_app():
    app = Flask(__name__)
    cors = CORS(app)
    app.config["CORS_HEADERS"] = "Content-Type"

    with open('tree_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)


        
    #Nueva sección
    path = "modelos/anntr.model"
    modelo = joblib.load(path)


    #Fin sección
    #Tercer modelo solo IRCA
    with open('modelos/tree_modelOnlyIrca.pkl', 'rb') as model_file:
        modelIRCA = pickle.load(model_file)


    #FUNCIÓN FINAL
    @app.route("/", methods=["GET", "POST"])
    def final():
        if request.method == 'POST':
            try:
                data = request.json['data']
            
                X = pd.DataFrame(data)
            
                predictions_modelo = modelo.predict(X)
                salida = predictions_modelo[0]

                data_modelIRCA = {"features": [[salida]]}
                features_modelIRCA = data_modelIRCA["features"]

                resultadoFinal_modeloIRCA = modelIRCA.predict(features_modelIRCA)

                # Crear un diccionario con los resultados de ambos modelos
                resultados = {
                    "predictions_modelo": predictions_modelo.tolist(),
                    "resultadoFinal_modeloIRCA": resultadoFinal_modeloIRCA.tolist()
                }

                return jsonify(resultados)

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        elif request.method == 'GET':
            
            return "¡Modelo final prueba!", 200

    return app

if __name__ == "__main__":
    app = crear_app()
    app.run(debug=True)

