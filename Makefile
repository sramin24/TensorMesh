.PHONY: proto

# Regenerates the gRPC Python bindings from protocol.proto. Generated files
# (protocol_pb2.py, protocol_pb2_grpc.py) are gitignored — this is a required
# manual/pre-build step, not something you edit by hand or commit.
proto:
	python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. tensormesh/network/proto/protocol.proto
