import time
import gzip
import zlib
import lz4.frame
import msgpack
from google.protobuf import descriptor_pool, descriptor_pb2, message_factory
import os
import sys


from google.protobuf import descriptor_pool, descriptor_pb2, message_factory


def build_proto_message():
    """Dynamically create a protobuf message with a bytes field."""
    file_desc_proto = descriptor_pb2.FileDescriptorProto()
    file_desc_proto.name = "data.proto"

    msg_type = file_desc_proto.message_type.add()
    msg_type.name = "DataMessage"

    field = msg_type.field.add()
    field.name = "data"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_BYTES

    pool = descriptor_pool.Default()
    pool.Add(file_desc_proto)

    msg_desc = pool.FindMessageTypeByName("DataMessage")

    # New protobuf API
    return message_factory.GetMessageClass(msg_desc)


def benchmark(name, compress_fn, decompress_fn, data):
    # Compress
    start = time.perf_counter()
    compressed = compress_fn(data)
    comp_time = time.perf_counter() - start

    # Decompress
    start = time.perf_counter()
    decompressed = decompress_fn(compressed)
    decomp_time = time.perf_counter() - start

    # Verify
    if decompressed != data:
        raise RuntimeError(f"{name} decompression failed integrity check!")

    ratio = (len(data) / len(compressed))

    return {
        "algorithm": name,
        "compressed_size": len(compressed),
        "compression_ratio": ratio,
        "compression_time": comp_time,
        "decompression_time": decomp_time,
    }


def main(file_path):
    with open(file_path, "rb") as f:
        data = f.read()

    print(f"File: {file_path}")
    print(f"Original size: {len(data):,} bytes\n")

    results = []

    # Gzip
    results.append(
        benchmark(
            "Gzip",
            lambda d: gzip.compress(d),
            lambda d: gzip.decompress(d),
            data,
        )
    )

    # Deflate (zlib)
    results.append(
        benchmark(
            "Deflate (zlib)",
            lambda d: zlib.compress(d),
            lambda d: zlib.decompress(d),
            data,
        )
    )

    # LZ4
    results.append(
        benchmark(
            "LZ4",
            lambda d: lz4.frame.compress(d),
            lambda d: lz4.frame.decompress(d),
            data,
        )
    )

    # MessagePack
    results.append(
        benchmark(
            "MessagePack",
            lambda d: msgpack.packb(d),
            lambda d: msgpack.unpackb(d),
            data,
        )
    )

    # Protocol Buffers
    ProtoMessage = build_proto_message()

    def proto_compress(d):
        msg = ProtoMessage()
        msg.data = d
        return msg.SerializeToString()

    def proto_decompress(b):
        msg = ProtoMessage()
        msg.ParseFromString(b)
        return msg.data

    results.append(
        benchmark(
            "Protocol Buffers",
            proto_compress,
            proto_decompress,
            data,
        )
    )

    # Print results
    print(f"{'Algorithm':<18} {'Ratio':<10} {'Comp Time (s)':<15} {'Decomp Time (s)':<15} {'Size'}")
    print("-" * 70)

    for r in results:
        print(
            f"{r['algorithm']:<18} "
            f"{r['compression_ratio']:<10.4f} "
            f"{r['compression_time']:<15.6f} "
            f"{r['decompression_time']:<15.6f} "
            f"{r['compressed_size']:,}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python benchmark_compression.py <test_file>")
        sys.exit(1)

    main(sys.argv[1])