# 多阶段构建: 编译 + 最小运行镜像
FROM golang:1.22 AS builder
WORKDIR /app
COPY . .
RUN go mod tidy && CGO_ENABLED=0 go build -o training-operator main.go

FROM alpine:3.19
COPY --from=builder /app/training-operator /training-operator
ENTRYPOINT ["/training-operator"]
