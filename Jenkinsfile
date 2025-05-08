pipeline {
    agent any  // Utilise n'importe quel agent (machine) disponible

    stages {
        stage('Checkout') {
            steps {   
                // Cloner le dépôt Git
                git credentialsId: 'bcbfb519-cbf9-4eb8-a522-dd9008ffe6ac', url: 'https://github.com/Marie-2000/MLOPS.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                // Installer les dépendances (ici pour un projet Python)
                script {
                    // Sur Windows, on utilise 'bat' pour les commandes batch
                    bat 'pip install -r requirements.txt'
                }
            }
        }

        stage('Test') {
            steps {
                // Lancer les tests (ici, avec pytest)
                script {
                    // Utiliser 'bat' pour exécuter pytest sous Windows
                    bat 'pytest --maxfail=1 --disable-warnings -q'
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
