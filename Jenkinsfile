pipeline {
    agent any

    environment {
        WORKDIR = '/var/jenkins_home/workspace/fastapi-app'        // FastAPI 프로젝트 경로
        COMPOSE_SERVICE = 'fast-api'                // docker-compose 서비스명
        IMAGE_NAME = 'fastapi-app:latest'
    }

    stages {

        stage('Prepare') {
            steps {
                echo "🔄 Cloning FastAPI Repository..."
                dir("${WORKDIR}") {
                    git branch: 'main', url: 'https://github.com/ShootPointer/ShootPointer_OpenCV.git'
                }
            }
            post {
                success { sh 'echo "✅ Successfully Cloned Repository"' }
                failure { sh 'echo "❌ Failed to Clone Repository"' }
            }
        }

        stage('Replace .env File') {
            steps {
                echo "🔐 Copying Secret .env file..."
                script {
                    withCredentials([file(credentialsId: 'SECRET_FILE_OPEN_CV', variable: 'secretFile')]) {
                        sh 'cp $secretFile ${WORKDIR}/.env'
                    }
                }
            }
            post {
                success { sh 'echo "✅ Successfully Replaced .env File"' }
                failure { sh 'echo "❌ Failed to Replace .env File"' }
            }
        }

        stage('Build Docker Image') {
            steps {
                dir("${WORKDIR}") {
                    sh 'echo "🐳 Building FastAPI Docker Image..."'
                    sh '''
                    docker build -t ${IMAGE_NAME} .
                    '''
                }
            }
            post {
                success { sh 'echo "✅ Successfully Built Docker Image"' }
                failure { sh 'echo "❌ Failed to Build Docker Image"' }
            }
        }

        stage('Deploy FastAPI & AI Worker Containers') {
    steps {
        dir("${WORKDIR}") {
            sh 'echo "🚀 Deploying FastAPI + AI Worker Containers..."'
            script {
                def runningFastAPI = sh(script: "docker ps -q -f name=fast-api", returnStdout: true).trim()
                def runningWorker = sh(script: "docker ps -q -f name=ai_worker_simulator", returnStdout: true).trim()

                if (runningFastAPI) {
                    sh "docker stop fast-api || true"
                    sh "docker rm -f fast-api || true"
                }

                if (runningWorker) {
                    sh "docker stop ai_worker_simulator || true"
                    sh "docker rm -f ai_worker_simulator || true"
                }

                sh 'echo "🚀 Starting Both Services..."'
                sh "docker-compose up -d --build fast-api ai_worker"
            }
        }
    }
    post {
        success { sh 'echo "✅ Successfully Deployed FastAPI + AI Worker"' }
        failure { sh 'echo "❌ Deployment Failed (check logs)"' }
    }
}

        stage('Clean Up Old Images') {
            steps {
                sh 'echo "🧹 Cleaning Up Unused Docker Images..."'
                sh 'docker image prune -f || true'
            }
            post {
                success { sh 'echo "✅ Successfully Cleaned Up Old Images"' }
                failure { sh 'echo "❌ Failed to Clean Up Images"' }
            }
        }
    }

    post {
        success {
            sh 'echo "🎉 FastAPI Deployment Completed Successfully!"'
        }
        failure {
            sh 'echo "💥 FastAPI Deployment Failed. Check the Logs Above."' 
        }
    }
}
