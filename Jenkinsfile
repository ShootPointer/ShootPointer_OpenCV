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

        stage('Deploy FastAPI Container') {
            steps {
                dir("${WORKDIR}") {
                    sh 'echo "🚀 Deploying FastAPI Container..."'
                    script {
                        def running = sh(script: "docker ps -q -f name=${COMPOSE_SERVICE}", returnStdout: true).trim()

                        if (running) {
                            sh 'echo "🛑 Stopping and Removing Existing Container..."'
                            sh "docker stop ${COMPOSE_SERVICE} || true"
                            sh "docker rm -f ${COMPOSE_SERVICE} || true"
                        }

                        sh 'echo "🚀 Starting New FastAPI Container..."'
                        sh "docker-compose up -d --build ${COMPOSE_SERVICE}"
                    }
                }
            }
            post {
                success { sh 'echo "✅ Successfully Deployed FastAPI Container"' }
                failure { sh 'echo "❌ Failed to Deploy FastAPI Container"' }
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
