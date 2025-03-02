pipeline {
    agent any
    environment {
        DOCKER_HUB_USERNAME = 'essaidimaryem'  // Your Docker Hub username
        DOCKER_HUB_PASSWORD = credentials('DOCKER_CREDENTIALS')  // Jenkins credential ID
        MLFLOW_TRACKING_URI = 'http://localhost:5000'
        JENKINS = 'true'
    }
    stages {
        stage('Checkout') {
            steps {
                git branch: 'master',  // Changed from 'main' to 'master'
                    credentialsId: 'ML_Pipeline',
                    url: 'https://github.com/essaidimaryem8/ML-Pipeline.git'
            }
        }
        stage('Set Up Python') {
            steps {
                sh 'python3 -m venv venv'
                sh '''
                    . venv/bin/activate
                    python3 -m pip install --upgrade pip
                    python3 -m pip install setuptools
                    python3 -m pip install python-multipart
                    python3 -m pip install -r backend/requirements.txt -r frontend/requirements.txt
                '''
            }
        }
        stage('Code Quality') {
            steps {
                sh '''
                    . venv/bin/activate
                    pycodestyle backend/main.py backend/model_pipeline/preprocessing.py backend/model_pipeline/training.py backend/model_pipeline/evaluation.py backend/model_pipeline/io.py backend/tests/test_main.py --max-line-length=120 --ignore=E203,E266,E501,W503
                '''
            }
        }
        stage('Run Tests') {
            steps {
                dir('backend') {
                    sh '''
                        . ../venv/bin/activate
                        export PYTHONPATH=$PYTHONPATH:..
                        export MLFLOW_TRACKING_URI=file:///tmp/mlflow-tests
                        export TESTING=true # Set TESTING to true to skip MLflow during tests
                        pytest tests/test_main.py -v
                    '''
                }
            }
        }
        stage('Start MLflow Server') {
            steps {
                dir('backend') {
                    sh '''
                        . ../venv/bin/activate
                        nohup mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns &
                        sleep 15
                    '''
                }
            }
        }
        stage('Build Docker Images') {
            steps {
                script {
                    // Login to Docker Hub
                    sh 'echo $DOCKER_HUB_PASSWORD_PSW | docker login -u $DOCKER_HUB_USERNAME --password-stdin'

                    // Build backend image
                    sh "docker build -t ${DOCKER_HUB_USERNAME}/ml-pipeline-backend:latest -f backend/Dockerfile ./backend"
                    // Build frontend image
                    sh "docker build -t ${DOCKER_HUB_USERNAME}/ml-pipeline-frontend:latest -f frontend/Dockerfile ./frontend"
                }
            }
        }
        stage('Push Docker Images') {
            steps {
                script {
                    // Ensure Docker Hub login
                    sh 'echo $DOCKER_HUB_PASSWORD_PSW | docker login -u $DOCKER_HUB_USERNAME --password-stdin'

                    // Push backend image
                    sh "docker push ${DOCKER_HUB_USERNAME}/ml-pipeline-backend:latest"
                    // Push frontend image
                    sh "docker push ${DOCKER_HUB_USERNAME}/ml-pipeline-frontend:latest"
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
                // Remove existing mlflow.db to avoid UNIQUE constraint errors
                sh 'docker exec backend rm -f /app/mlflow.db'
                // Run pipeline steps sequentially with logging
                sh '''
                    docker exec backend bash -c "export MLFLOW_TRACKING_URI=http://mlflow:5000 && export PYTHONPATH=/app && python /app/main.py --prepare_data 2>&1 | tee /app/prepare_data.log"
                '''
                sh '''
                    docker exec backend bash -c "export MLFLOW_TRACKING_URI=http://mlflow:5000 && export PYTHONPATH=/app && python /app/main.py --train 2>&1 | tee /app/train.log"
                '''
                sh '''
                    docker exec backend bash -c "export MLFLOW_TRACKING_URI=http://mlflow:5000 && export PYTHONPATH=/app && python /app/main.py --evaluate 2>&1 | tee /app/evaluate.log"
                '''
                sh '''
                    docker exec backend bash -c "export MLFLOW_TRACKING_URI=http://mlflow:5000 && export PYTHONPATH=/app && python /app/main.py --retrain 2>&1 | tee /app/retrain.log"
                '''
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
            // Change directory to where docker-compose.yml is located
            dir('/var/lib/jenkins/workspace/ML-Pipeline') {
                sh 'docker-compose down || true'  // Ignore errors if docker-compose.yml is not found
            }
            sh 'docker system prune -f'
        }
        success {
            echo 'Pipeline completed successfully!'
            script {
                try {
                    withCredentials([string(credentialsId: 'sender-email', variable: 'SENDER_EMAIL'),
                                     string(credentialsId: 'recipient-email', variable: 'RECIPIENT_EMAIL')]) {
                        mail to: "${RECIPIENT_EMAIL}",
                             subject: "Pipeline Success: ML Pipeline",
                             body: "The ML Pipeline job completed successfully on ${env.BUILD_URL}. Docker images are pushed to ${DOCKER_HUB_USERNAME}/ml-pipeline-backend:latest and ${DOCKER_HUB_USERNAME}/ml-pipeline-frontend:latest."
                    }
                } catch (Exception e) {
                    echo "Failed to send success email: ${e.getMessage()}"
                }
            }
        }
        failure {
            echo 'Pipeline failed!'
            script {
                try {
                    withCredentials([string(credentialsId: 'sender-email', variable: 'SENDER_EMAIL'),
                                     string(credentialsId: 'recipient-email', variable: 'RECIPIENT_EMAIL')]) {
                        mail to: "${RECIPIENT_EMAIL}",
                             subject: "Pipeline Failure: ML Pipeline",
                             body: "The ML Pipeline job failed on ${env.BUILD_URL}. Check logs for details."
                    }
                } catch (Exception e) {
                    echo "Failed to send failure email: ${e.getMessage()}"
                }
            }
        }
    }
}
