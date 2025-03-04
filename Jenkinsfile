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
                    // Check if requirements.txt (in root) or backend/model_pipeline/ has changed
                    def changes = sh(script: "git diff --name-only HEAD^ HEAD", returnStdout: true).trim()
                    if (changes.contains('requirements.txt') || changes.contains('backend/model_pipeline/')) {
                        sh '''
                            docker build -f Dockerfile.dependencies -t essaidimaryem/ml-project:latest .
                            echo $DOCKER_CREDENTIALS_PSW | docker login -u $DOCKER_CREDENTIALS_USR --password-stdin
                            docker push essaidimaryem/ml-project:latest
                        '''
                    } else {
                        echo "No changes in requirements.txt or backend/model_pipeline/, skipping dependencies image build."
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
                            -e PYTHONPATH=/app:/app/backend \
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
                    sh 'docker build -f backend/Dockerfile -t essaidimaryem/ml-pipeline-backend:latest ./backend'
                    sh 'docker build -f frontend/Dockerfile -t essaidimaryem/ml-pipeline-frontend:latest ./frontend'
                }
            }
        }
        stage('Push Docker Images') {
            steps {
                script {
                    sh '''
                        echo $DOCKER_CREDENTIALS_PSW | docker login -u $DOCKER_CREDENTIALS_USR --password-stdin
                        # Push images in parallel
                        docker push essaidimaryem/ml-pipeline-backend:latest &
                        docker push essaidimaryem/ml-pipeline-frontend:latest &
                        wait
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
                    # Wait for MLflow server to be ready
                    for i in {1..30}; do
                        if curl -s http://localhost:5001; then
                            echo "MLflow server is up!"
                            break
                        fi
                        echo "Waiting for MLflow server... ($i/30)"
                        sleep 2
                    done
                    if ! curl -s http://localhost:5001; then
                        echo "MLflow server did not start in time"
                        exit 1
                    fi
                '''
                // Run the pipeline steps
                sh '''
                    docker exec backend python /app/main.py --prepare_data --train --evaluate --retrain
                '''
            }
        }
    }
    post {
        always {
            node('') {  // Use empty label to run on any available agent
                dir(env.WORKSPACE) {
                    sh '''
                        # Clear mlruns directory to prevent permission issues in future runs
                        rm -rf mlruns || true
                        docker-compose down || true
                    '''
                }
                sh '''
                    docker system prune -f
                '''
                script {
                    withCredentials([string(credentialsId: 'sender-email', variable: 'SENDER_EMAIL'), string(credentialsId: 'recipient-email', variable: 'RECIPIENT_EMAIL')]) {
                        mail to: "${RECIPIENT_EMAIL}",
                             subject: "Pipeline ${currentBuild.result == 'SUCCESS' ? 'Succeeded' : 'Failed'}: ${env.JOB_NAME} Build #${env.BUILD_NUMBER}",
                             body: "The pipeline ${env.JOB_NAME} Build #${env.BUILD_NUMBER} has ${currentBuild.result == 'SUCCESS' ? 'succeeded' : 'failed'}. Check the logs at ${env.BUILD_URL}"
                    }
                }
            }
        }
    }
}
