pipeline {
    agent any
    environment {
        DOCKER_IMAGE = 'essaidimaryem/ml-project'
        DOCKER_TAG = 'latest'
        PATH = "/usr/local/bin:/usr/bin:/bin:${PATH}"
    }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Setup Python Environment') {
            steps {
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    python3 -m pip install --upgrade pip
                    python3 -m pip install pycodestyle pytest -r requirements.txt
                '''
            }
        }
        stage('Code Quality') {
            steps {
                sh '''
                    . .venv/bin/activate
                    pycodestyle backend/main.py backend/model_pipeline.py tests/test_main.py --max-line-length=120 --ignore=E203,E266,E501,W503
                '''
            }
        }
        stage('Run Tests') {
            steps {
                sh '''
                    . .venv/bin/activate
                    pytest tests/test_main.py
                '''
            }
        }
        stage('Prepare Data') {
            steps {
                sh '''
                    . .venv/bin/activate
                    python backend/main.py --prepare_data
                '''
            }
        }
        stage('Train Model') {
            steps {
                sh '''
                    . .venv/bin/activate
                    python backend/main.py --train
                '''
            }
        }
        stage('Evaluate Model') {
            steps {
                sh '''
                    . .venv/bin/activate
                    python backend/main.py --evaluate
                '''
            }
        }
        stage('Build & Push Docker Image') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'DOCKER_CREDENTIALS', 
                                                passwordVariable: 'DOCKER_PASSWORD', 
                                                usernameVariable: 'DOCKER_USERNAME')]) {
                    sh '''
                        docker build -t $DOCKER_IMAGE:$DOCKER_TAG .
                        echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
                        docker push $DOCKER_IMAGE:$DOCKER_TAG
                    '''
                }
            }
        }
    }
    post {
        always {
            cleanWs()
        }
    }
}
