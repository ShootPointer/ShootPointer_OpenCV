pipeline {
    agent any

    environment {
        // 환경 변수
        DOCKER_COMPOSE_PATH = "/home/ubuntu/fastapi-app"  // 서버 내 docker-compose 위치
        CONTAINER_NAME = "fastapi-app"
        GIT_REPO = "https://github.com/USERNAME/REPOSITORY.git"
        BRANCH = "main"
    }

    stages {
        stage('Checkout') {
            steps {
                echo "✅ Checking out source code..."
                git branch: "${BRANCH}", url: "${GIT_REPO}"
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "🐳 Building Docker image..."
                sh 'docker-compose build --no-cache'
            }
        }

        stage('Deploy') {
            steps {
                echo "🚀 Deploying container..."
                // 기존 컨테이너 중단 및 제거
                sh '''
                cd ${DOCKER_COMPOSE_PATH}
                docker-compose down
                docker-compose up -d --build
                '''
            }
        }

        stage('Cleanup') {
            steps {
                echo "🧹 Removing dangling images..."
                sh 'docker image prune -f'
            }
        }
    }

    post {
        success {
            echo "🎉 Deployment successful!"
        }
        failure {
            echo "❌ Deployment failed. Please check logs."
        }
    }
}
