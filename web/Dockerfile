# Two stages, because the runner needs the build's output and none of its
# toolchain.
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .

# NEXT_PUBLIC_* is compiled into the browser bundle, not read from the
# environment at run time - the browser never sees this container's env at all.
# So they are build args: changing either one means rebuilding this image.
ARG NEXT_PUBLIC_WS_URL=ws://localhost:8000
ARG NEXT_PUBLIC_WS_TOKEN=
ENV NEXT_PUBLIC_WS_URL=$NEXT_PUBLIC_WS_URL NEXT_PUBLIC_WS_TOKEN=$NEXT_PUBLIC_WS_TOKEN
RUN npm run build

FROM node:22-alpine
WORKDIR /app
# HOSTNAME is what the standalone server binds; next's default is 0.0.0.0 but
# it reads this, and a container that binds loopback answers nobody.
ENV NODE_ENV=production HOSTNAME=0.0.0.0 PORT=3000
COPY --from=build /app/public ./public
# standalone is next's own trace of what the build actually imported, so this
# is the app plus the node_modules it really uses, rather than all of them.
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
USER node
EXPOSE 3000
CMD ["node", "server.js"]
