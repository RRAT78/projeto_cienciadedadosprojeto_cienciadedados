# projeto_cienciadedadosprojeto_cienciadedados
#exemplo: !git clone [https://github.com/RRAT78/projeto_cienciadedadosprojeto_cienciadedados.git]
!git clone <https://github.com/RRAT78/projeto_cienciadedadosprojeto_cienciadedados.git> #substitua esse final pelo link do seu projeto
conteudo_codigo = '''
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threading import Thread

# Inicializar o aplicativo Flask
app = Flask(__name__)

# Carregar o dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir em dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar e treinar o modelo
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)


@app.route('/')
def home():
    return "API de Classificação do Iris Dataset"

# Rota para fazer previsões
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Obter os parâmetros de entrada da requisição
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))

        # Carregar o modelo e o escalador
        model = joblib.load('iris_model.pkl')
        scaler = joblib.load('scaler.pkl')

        # Normalizar os dados de entrada
        input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])

        # Fazer a previsão
        prediction = model.predict(input_data)

        # Mapear a previsão para o nome da classe
        iris_target_names = iris.target_names
        predicted_class = iris_target_names[prediction[0]]

        # Retornar a previsão como resposta JSON
        return jsonify({
            'predicted_class': predicted_class
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Function to run the Flask app
def run():
    app.run(port=5000, debug=False, use_reloader=False)  # Prevents reloading, works better in notebooks

# Start the Flask app in a background thread
thread = Thread(target=run)
thread.start()
'''
nome_arquivo = "app.py"

# Abrindo o arquivo no modo de escrita
with open(nome_arquivo, "w") as f:
    f.write(conteudo_codigo)

print(f"Código Python foi salvo em {nome_arquivo}")
!mv app.py projeto_cienciadedados
dependencies = """
flask==2.3.2
scikit-learn==1.2.1
"""

# Specify the filename for the requirements file
requirements_filename = "requirements.txt"

# Write the dependencies to the requirements.txt file
with open(requirements_filename, "w") as f:
    f.write(dependencies)
    !mv requirements.txt projeto_cienciadedados
    %cd projeto_cienciadedados
    !git config --global user.name RRAT78 ## seu nome
!git config --global user.email lais.silva2@uscsonline.com.br
!git status

!git add .
 #criando um commit com a mensagem determinada
 #mude a mensagem se quiser
!git commit -m "Meu Primeiro Commit" .
