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
                    docker-compose build mlflow
                    docker-compose up -d mlflow
                '''
            }
        }
        stage('Build Docker Images') {
            steps {
                script {
                    // Changed context to project root
                    sh 'docker build -f backend/Dockerfile -t essaidimaryem/ml-pipeline-backend:latest .'
                    sh 'docker build -f frontend/Dockerfile -t essaidimaryem/ml-pipeline-frontend:latest ./frontend'
                }
            }
        }
        stage('Push Docker Images') {
            steps {
                script {
                    sh '''
                        echo $DOCKER_CREDENTIALS_PSW | docker login -u $DOCKER_CREDENTIALS_USR --password-stdin

                        # Retry function for docker push with exponential backoff
                        retry_push() {
                            local image=$1
                            local max_attempts=3
                            local attempt=1
                            local delay=5
                            while [ $attempt -le $max_attempts ]; do
                                echo "Pushing $image (attempt $attempt/$max_attempts)..."
                                if docker push $image; then
                                    echo "Successfully pushed $image"
                                    return 0
                                else
                                    echo "Failed to push $image"
                                    if [ $attempt -eq $max_attempts ]; then
                                        echo "Max attempts reached for $image, giving up."
                                        return 1
                                    fi
                                    sleep $delay
                                    attempt=$((attempt + 1))
                                    delay=$((delay * 2))
                                fi
                            done
                        }

                        # Push images sequentially with retries
                        retry_push essaidimaryem/ml-project:latest || exit 1
                        retry_push essaidimaryem/ml-pipeline-backend:latest || exit 1
                        retry_push essaidimaryem/ml-pipeline-frontend:latest || exit 1
                    '''
                }
            }
        }
        stage('Run Pipeline') {
            steps {
                sh '''
                    # Check and free port 5001 for MLflow
                    PORT=5001
                    if lsof -i:$PORT; then
                        echo "Port $PORT is in use, attempting to free it..."
                        PID=$(lsof -i:$PORT -t)
                        kill -9 $PID
                        echo "Port $PORT has been freed."
                    else
                        echo "Port $PORT is not in use."
                    fi
                    # Check and free port 8000 for backend
                    PORT=8000
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
                    # Remove the backend-data volume to ensure it is recreated and initialized
                    docker volume rm ml-pipeline_backend-data || true
                    docker-compose build backend mlflow elasticsearch kibana frontend
                    docker-compose up -d backend mlflow elasticsearch kibana frontend
                    # Debug: Check initial logs for MLflow and backend
                    echo "Checking initial MLflow server logs..."
                    docker-compose logs mlflow
                    echo "Checking initial backend service logs..."
                    docker-compose logs backend
                    echo "Checking initial frontend service logs..."
                    docker-compose logs frontend
                    # Debug: Verify data files in /app/data
                    echo "Checking data files in /app/data..."
                    docker-compose exec backend ls -l /app/data
                    # Wait for MLflow server to be ready
                    for i in {1..60}; do
                        if curl -s http://localhost:5001; then
                            echo "MLflow server is up!"
                            break
                        fi
                        echo "Waiting for MLflow server... ($i/60)"
                        sleep 2
                    done
                    if ! curl -s http://localhost:5001; then
                        echo "MLflow server did not start in time, checking final logs..."
                        docker-compose logs mlflow
                        exit 1
                    fi
                    # Wait for backend service to be ready
                    echo "Waiting for backend service to be ready..."
                    for i in {1..30}; do
                        if docker-compose ps backend | grep "Up"; then
                            echo "Backend service is up!"
                            break
                        fi
                        echo "Waiting for backend service... ($i/30)"
                        sleep 2
                    done
                    if ! docker-compose ps backend | grep "Up"; then
                        echo "Backend service did not start in time, checking final logs..."
                        docker-compose logs backend
                        exit 1
                    fi
                    # Wait for frontend service to be ready
                    echo "Waiting for frontend service to be ready..."
                    for i in {1..30}; do
                        if docker-compose ps frontend | grep "Up"; then
                            echo "Frontend service is up!"
                            break
                        fi
                        echo "Waiting for frontend service... ($i/30)"
                        sleep 2
                    done
                    if ! docker-compose ps frontend | grep "Up"; then
                        echo "Frontend service did not start in time, checking final logs..."
                        docker-compose logs frontend
                        exit 1
                    fi
                '''
                sh '''
                    docker-compose exec backend python /app/main.py --prepare_data --train --evaluate --retrain
                '''
            }
        }
    }
    post {
        always {
            node('') {  // Use empty label to run on any available agent
                dir(env.WORKSPACE) {
                    sh '''
                        # Clear mlruns and pytest cache directories to prevent permission issues in future runs
                        rm -rf mlruns || true
                        rm -rf backend/.pytest_cache || true
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
