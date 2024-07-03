FROM node:alpine

WORKDIR '/app'

COPY .npmrc package.json ./
RUN npm install
COPY . .

# Expose the port that the app runs on
EXPOSE 5173

CMD ["npm", "run", "dev"] 