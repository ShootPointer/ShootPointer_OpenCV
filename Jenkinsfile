pipeline {
    agent any

    environment {
        WORKDIR = '/home/ubuntu/fastapi-app'       // FastAPI 프로젝트 경로
        COMPOSE_SERVICE = 'fast-api'               // docker-compose 서비스명 변경됨
        IMAGE_NAME = 'fastapi-app:latest'
    }

    stages {

        stage('Checkout Repository') {
            steps {
                echo "🔄 Cloning FastAPI repository..."
                dir("${WORKDIR}") {
                    git branch: 'main', url: 'https://github.com/ShootPointer/ShootPointer_OpenCV.git'
                }
            }
            post {
                success { echo "✅ Repository cloned successfully." }
                failure { echo "❌ Failed to clone repository." }
            }
        }

        stage('Build Docker Image') {
            steps {
                dir("${WORKDIR}") {
                    echo "🐳 Building FastAPI Docker image..."
                    sh '''
                    docker build -t ${IMAGE_NAME} .
                    '''
                }
            }
            post {
                success { echo "✅ Docker image built successfully." }
                failure { echo "❌ Docker build failed." }
            }
        }

        stage('Deploy FastAPI Container') {
            steps {
                dir("${WORKDIR}") {
                    echo "🚀 Deploying FastAPI container..."
                    script {
                        def running = sh(script: "docker ps -q -f name=${COMPOSE_SERVICE}", returnStdout: true).trim()
                        if (running) {
                            echo "🛑 Stopping and removing existing container..."
                            sh "docker stop ${COMPOSE_SERVICE} || true"
                            sh "docker rm -f ${COMPOSE_SERVICE} || true"
                        }

                        echo "🚀 Starting new FastAPI container..."
                        sh "docker-compose up -d --build ${COMPOSE_SERVICE}"
                    }
                }
            }
            post {
                success { echo "✅ FastAPI container deployed successfully." }
                failure { echo "❌ Deployment failed." }
            }
        }

        stage('Clean Up Old Images') {
            steps {
                echo "🧹 Cleaning up old images..."
                sh 'docker image prune -f || true'
            }
        }
    }

    post {
        success {
            echo "🎉 FastAPI Deployment completed successfully!"
        }
        failure {
            echo "💥 Deployment failed. Check the logs above."
        }
    }
}
