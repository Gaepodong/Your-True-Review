로컬에서 실행시, 아래 순서대로 실행해주세요.

1. virtualenv 가상환경 활성화
    `source .venv/bin/activate`
2. 환경변수 입력
    `export FLASK_APP="true_review"`
    `export FLASK_ENV="devlopment"`
	`export API_SERVER="http://13.124.240.211:55637/predict"`
3. 의존성 파일 설치
    `pip install -r requirement.txt`
4. 플라스크 실행
    `flask run`


아시겠지만 virtualenv 가상환경 비활성화는 `deactivate` 입력하시면 됩니다.
