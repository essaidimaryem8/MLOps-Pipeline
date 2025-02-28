pipeline {
    agent any
    environment {
        DOCKER_HUB_USERNAME = 'essaidimaryem'
        DOCKER_HUB_PASSWORD = credentials('DOCKER_CREDENTIALS')
        MLFLOW_TRACKING_URI = 'http://localhost:5000'
        JENKINS = 'true'
    }
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/essaidimaryem8/ML-Pipeline.git'
            }
        }
        stage('Set Up Python') {
            steps {
                sh 'python3 -m venv venv'
                sh '. venv/bin/activate'
                sh 'pip install --upgrade pip'
                sh 'pip install -r backend/requirements.txt -r frontend/requirements.txt'
            }
        }
        stage('Code Quality') {
            steps {
                sh 'pycodestyle backend/main.py backend/model_pipeline/preprocessing.py backend/model_pipeline/training.py backend/model_pipeline/evaluation.py backend/model_pipeline/io.py tests/test_main.py --max-line-length=120 --ignore=E203,E266,E501,W503'
            }
        }
        stage('Run Tests') {
            steps {
                dir('backend') {
                    sh 'pytest tests/test_main.py -v'
                }
            }
        }
        stage('Start MLflow Server') {
            steps {
                dir('backend') {
                    sh 'nohup mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns &'
                    // Wait for server to start
                    sh 'sleep 15'
                }
            }
        }
        stage('Build Docker Images') {
            steps {
                script {
                    docker.build("${DOCKER_HUB_USERNAME}/ml-pipeline-backend:latest", './backend')
                    docker.build("${DOCKER_HUB_USERNAME}/ml-pipeline-frontend:latest", './frontend')
                }
            }
        }
        stage('Push Docker Images') {
            steps {
                script {
                    docker.withRegistry('https://index.docker.io/v1/', 'docker-hub-credentials') {
                        docker.image("${DOCKER_HUB_USERNAME}/ml-pipeline-backend:latest").push()
                        docker.image("${DOCKER_HUB_USERNAME}/ml-pipeline-frontend:latest").push()
                    }
                }
            }
        }
        stage('Run Pipeline') {
            steps {
                sh 'docker-compose down'
                sh 'docker-compose build backend mlflow elasticsearch kibana'
                sh 'docker-compose up -d backend mlflow elasticsearch kibana'
                // Wait for services to start
                sh 'sleep 15'
                sh 'docker exec backend python /app/main.py --prepare_data --train --evaluate --retrain'
                sh 'mlflow experiments --experiment-name GBM_Experiment'  // Verify tracking
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying to production (simulated)'
                echo "Docker images pushed: ${DOCKER_HUB_USERNAME}/ml-pipeline-backend:latest and ${DOCKER_HUB_USERNAME}/ml-pipeline-frontend:latest"
                echo "You can now pull and run these images locally with Docker Compose."
            }
        }
    }
    post {
        always {
            sh 'docker-compose down'
            sh 'docker system prune -f'
        }
        success {
            echo 'Pipeline completed successfully!'
            withCredentials([string(credentialsId: 'sender-email', variable: 'SENDER_EMAIL'),
                             string(credentialsId: 'recipient-email', variable: 'RECIPIENT_EMAIL')]) {
                mail to: "${RECIPIENT_EMAIL}",
                     subject: "Pipeline Success: ML Pipeline",
                     body: "The ML Pipeline job completed successfully on ${env.BUILD_URL}. Docker images are pushed to ${DOCKER_HUB_USERNAME}/ml-pipeline-backend:latest and ${DOCKER_HUB_USERNAME}/ml-pipeline-frontend:latest."
            }
        }
        failure {
            echo 'Pipeline failed!'
            withCredentials([string(credentialsId: 'sender-email', variable: 'SENDER_EMAIL'),
                             string(credentialsId: 'recipient-email', variable: 'RECIPIENT_EMAIL')]) {
                mail to: "${RECIPIENT_EMAIL}",
                     subject: "Pipeline Failure: ML Pipeline",
                     body: "The ML Pipeline job failed on ${env.BUILD_URL}. Check logs for details."
            }
        }
    }
}
