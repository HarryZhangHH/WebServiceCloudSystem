FROM nginx

# remove the default configuration file from public nginx image
RUN rm /etc/nginx/conf.d/default.conf

# copy the own nginx configuration file to container
COPY nginx.conf /etc/nginx/conf.d/
