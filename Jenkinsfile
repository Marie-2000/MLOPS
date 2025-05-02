pipeline {
    agent any  // Utilise n'importe quel agent (machine) disponible

    stages {
        stage('Checkout') {
            steps {
                // Cloner le dépôt Git
                git 'https://github.com/ton-utilisateur/ton-depot.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                // Installer les dépendances (ici pour un projet Python)
                script {
                    sh 'pip install -r requirements.txt'
                }
            }
        }

        stage('Test') {
            steps {
                // Lancer les tests (ici, avec pytest)
                script {
                    sh 'pytest'
                }
            }
        }

        stage('Deploy') {
            steps {
                // Déployer ton application (ici un simple print comme exemple)
                echo 'Déploiement de l’application'
            }
        }
    }

    post {
        always {
            echo 'Nettoyage après le pipeline'
        }
        success {
            echo 'Le pipeline a réussi'
        }
        failure {
            echo 'Le pipeline a échoué'
        }
    }
}
