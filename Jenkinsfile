pipeline {
    agent any
    environment {
        DOCKER_CREDENTIALS = credentials('DOCKER_CREDENTIALS')
        SENDER_EMAIL = credentials('sender-email')
        RECIPIENT_EMAIL = credentials('recipient-email')
    }
    stages {
        stage('Build Dependencies Image') {
            steps {
                script {
                    // Check if backend/requirements.txt or backend/model_pipeline/ has changed
                    def changes = sh(script: "git diff --name-only HEAD^ HEAD", returnStdout: true).trim()
                    if (changes.contains('backend/requirements.txt') || changes.contains('backend/model_pipeline/')) {
                        dir('backend') {
                            sh '''
                                docker build -f Dockerfile.dependencies -t essaidimaryem/ml-project:latest .
                                echo $DOCKER_CREDENTIALS_PSW | docker login -u $DOCKER_CREDENTIALS_USR --password-stdin
                                docker push essaidimaryem/ml-project:latest
                            '''
                        }
                    } else {
                        echo "No changes in backend/requirements.txt or backend/model_pipeline/, skipping dependencies image build."
                    }
                }
            }
        }
        stage('Code Quality') {
            steps {
                dir('backend') {
                    sh '''
                        docker run --rm \
                            -v $(pwd):/app \
                            essaidimaryem/ml-project:latest \
                            pycodestyle main.py model_pipeline/preprocessing.py model_pipeline/training.py model_pipeline/evaluation.py model_pipeline/io.py tests/test_main.py --max-line-length=120 --ignore=E127,E203,E266,E501,W503
                    '''
                }
            }
        }
        stage('Run Tests') {
            steps {
                dir('backend') {
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
                sh '''
                    docker-compose down
                    docker-compose up -d mlflow
                '''
            }
        }
        stage('Build Docker Images') {
            steps {
                script {
                    sh 'docker build -f Dockerfile.backend -t essaidimaryem/ml-pipeline-backend:latest ./backend'
                    sh 'docker build -f Dockerfile.frontend -t essaidimaryem/ml-pipeline-frontend:latest ./frontend'
                }
            }
        }
        stage('Push Docker Images') {
            steps {
                script {
                    sh '''
                        echo $DOCKER_CREDENTIALS_PSW | docker login -u $DOCKER_CREDENTIALS_USR --password-stdin
                        docker push essaidimaryem/ml-pipeline-backend:latest
                        docker push essaidimaryem/ml-pipeline-frontend:latest
                    '''
                }
            }
        }
        stage('Run Pipeline') {
            steps {
                // Free port 5001 if it's in use (updated port from previous step)
                sh '''
                    PORT=5001
                    if lsof -i:$PORT; then
                        echo "Port $PORT is in use, attempting to free it..."
                        PID=$(lsof -i:$PORT -t)
                        kill -9 $PID
                        echo "Port $PORT has been freed."
                    else
                        echo "Port $PORT is not in use."
                    fi
                '''
                sh '''
                    docker-compose down
                    docker-compose build backend mlflow elasticsearch kibana
                    docker-compose up -d backend mlflow elasticsearch kibana
                '''
                // Run the pipeline steps
                sh '''
                    docker exec backend python /app/main.py --prepare_data --train --evaluate --retrain
                '''
            }
        }
        stage('Deploy') {
            steps {
                echo "Deploying the application..."
                echo "Backend deployed at http://localhost:8000"
                echo "Frontend deployed at http://localhost:8501"
                echo "MLflow UI available at http://localhost:5001"
                echo "Elasticsearch available at http://localhost:9200"
                echo "Kibana available at http://localhost:5601"
            }
        }
    }
    post {
        always {
            dir(env.WORKSPACE) {
                sh '''
                    docker-compose down || true
                '''
            }
            sh '''
                docker system prune -f
            '''
            script {
                if (currentBuild.result == 'SUCCESS') {
                    echo "Pipeline succeeded!"
                } else {
                    echo "Pipeline failed!"
                }
                withCredentials([string(credentialsId: 'sender-email', variable: 'SENDER_EMAIL'), string(credentialsId: 'recipient-email', variable: 'RECIPIENT_EMAIL')]) {
                    mail to: "${RECIPIENT_EMAIL}",
                         subject: "Pipeline ${currentBuild.result == 'SUCCESS' ? 'Succeeded' : 'Failed'}: ${env.JOB_NAME} Build #${env.BUILD_NUMBER}",
                         body: "The pipeline ${env.JOB_NAME} Build #${env.BUILD_NUMBER} has ${currentBuild.result == 'SUCCESS' ? 'succeeded' : 'failed'}. Check the logs at ${env.BUILD_URL}"
                }
            }
        }
    }
}
