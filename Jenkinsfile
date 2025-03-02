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
                // No need to set up a virtual environment; dependencies are in the Docker image
                echo "Using essaidimaryem/ml-project image with preinstalled dependencies"
            }
        }
        stage('Code Quality') {
            steps {
                // Run pycodestyle inside a container using essaidimaryem/ml-project
                sh '''
                    docker run --rm \
                        -v $(pwd)/backend:/app \
                        essaidimaryem/ml-project:latest \
                        pycodestyle main.py model_pipeline/preprocessing.py model_pipeline/training.py model_pipeline/evaluation.py model_pipeline/io.py tests/test_main.py --max-line-length=120 --ignore=E127,E203,E266,E501,W503
                '''
            }
        }
        stage('Run Tests') {
            steps {
                dir('backend') {
                    // Run pytest inside a container using essaidimaryem/ml-project
                    sh '''
                        docker run --rm \
                            -v $(pwd)/..:/app \
                            -e PYTHONPATH=/app \
                            -e MLFLOW_TRACKING_URI=file:///tmp/mlflow-tests \
                            -e TESTING=true \
                            -w /app/backend \
                            essaidimaryem/ml-project:latest \
                            pytest tests/test_main.py -v
                    '''
                }
            }
        }
        stage('Start MLflow Server') {
            steps {
                dir('backend') {
                    // Start MLflow server inside a container using essaidimaryem/ml-project
                    sh '''
                        docker run -d \
                            --name mlflow-server \
                            -p 5000:5000 \
                            -v $(pwd)/..:/app \
                            -e PYTHONPATH=/app \
                            essaidimaryem/ml-project:latest \
                            mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///app/backend/mlflow.db --default-artifact-root /app/backend/mlruns
                    '''
                    // Wait for server to start
                    sh 'sleep 15'
                }
            }
        }
        stage('Build Docker Images') {
            steps {
                script {
                    // Login to Docker Hub
                    sh 'echo $DOCKER_HUB_PASSWORD_PSW | docker login -u $DOCKER_HUB_USERNAME --password-stdin'

                    // Build backend image (already using essaidimaryem/ml-project as base)
                    sh "docker build -t ${DOCKER_HUB_USERNAME}/ml-pipeline-backend:latest -f backend/Dockerfile ./backend"
                    // Build frontend image (already using essaidimaryem/ml-project as base)
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
                // Verify tracking by checking if GBM_Experiment exists
                sh '''
                    docker run --rm \
                        -v $(pwd)/backend/mlruns:/mlruns \
                        -e MLFLOW_TRACKING_URI=http://localhost:5000 \
                        essaidimaryem/ml-project:latest \
                        mlflow experiments list --view-type all | grep GBM_Experiment
                '''
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
