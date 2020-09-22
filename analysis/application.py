import json
from flask import Flask
from flask import request
from flask_cors import CORS
from faiss_similarity import FAISSIndex
from app_config import TEXT_EMBEDDING_LARGE_ALL, WIKI_LABELS

app = Flask(__name__)
CORS(app)

fi = FAISSIndex(TEXT_EMBEDDING_LARGE_ALL, WIKI_LABELS)
fi.build_index()


@app.route('/similarity/faiss/nn/<qnode>', methods=['GET'])
def faiss_nn(qnode):
    try:
        k = request.args.get("k", 5)
        results = fi.nearest_neighbors(qnode, k)
        return json.dumps(results), 200
    except Exception as e:
        return {'Error': str(e)}, 500


@app.route('/')
def is_alive():
    return 'hello from Qnode Similarity', 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6733)
