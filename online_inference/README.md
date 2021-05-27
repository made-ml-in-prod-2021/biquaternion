## Домашнее задание #2 Online inference

## запуск сервиса
### в python
`python3 app.py`
### в uvicorn
`uvicorn app:app --host=<HOSTNAME> [--port <PORT>] [--log-config conf/logging.conf]`

## работа с docker:

`docker login`  

собрать docker-образ  

`docker build -t biquaternion/online_inference:v2`  

запустить docker-образ  

`docker run -p 8000:8000 biquaternion/online_inference:v2 .`


