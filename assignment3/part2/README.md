# wscs_assignment3.2 - Container Orchestrations

## File Introduction
- login-deployment.yaml is used to deploy the login
- url-deployment.yaml is used to deploy the url-shortener
- login-service.yaml is used to launch the login service
- url-service.yaml is used to launch the url-shortener service
- nginx-ingress.yaml is used to perform reverse proxy in k8s(bonus implementation)

## Run(upload the file to master node)
```
kubectl apply -f login-deployment.yaml
kubectl apply -f url-deployment.yaml
kubectl apply -f login-service.yaml
kubectl apply -f url-service.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.1.1/deploy/static/provider/baremetal/deploy.yaml(bonus implementation)
kubectl get svc -n ingress-nginx(note down the external port mapped to 80)
kubectl apply -f nginx-ingress.yaml
kubectl get ingress(note down the ip address)
```