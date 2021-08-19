import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from scipy import misc
# Flask
# Jinja2
# numpy
# scikit-learn
# scipy
# virtualenv
# pillow


app = Flask(__name__)


# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


# 데이터 예측 처리
@app.route('/predict', methods=['GET'])
def make_prediction():
    if request.method == 'GET':

        # 원하는 값 받기
        picked = request.args.get('category_id', None)

        # 데이터 가져오기
        # for df in dflist :
        #     if df == picked : 
        #         picked = df




        # 입력 받은 이미지 예측
        prediction = model.fit(picked)  # 여기도 입력 받고 출력하는게 노출되어있다.



        # 결과 리턴
        return render_template('index.html', label=label)


if __name__ == '__main__':
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)
