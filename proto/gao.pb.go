// Code generated by protoc-gen-go.
// source: gao.proto
// DO NOT EDIT!

/*
Package proto is a generated protocol buffer package.

It is generated from these files:
	gao.proto

It has these top-level messages:
	GaoRequest
	GaoResponse
*/
package proto

import proto1 "github.com/golang/protobuf/proto"

import (
	context "golang.org/x/net/context"
	grpc "google.golang.org/grpc"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ context.Context
var _ grpc.ClientConn

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto1.Marshal

type GaoRequest struct {
	UserId int32 `protobuf:"varint,1,opt,name=user_id" json:"user_id,omitempty"`
}

func (m *GaoRequest) Reset()         { *m = GaoRequest{} }
func (m *GaoRequest) String() string { return proto1.CompactTextString(m) }
func (*GaoRequest) ProtoMessage()    {}

type GaoResponse struct {
	UserName string `protobuf:"bytes,1,opt,name=user_name" json:"user_name,omitempty"`
	Position string `protobuf:"bytes,2,opt,name=position" json:"position,omitempty"`
}

func (m *GaoResponse) Reset()         { *m = GaoResponse{} }
func (m *GaoResponse) String() string { return proto1.CompactTextString(m) }
func (*GaoResponse) ProtoMessage()    {}

func init() {
}

// Client API for GaoService service

type GaoServiceClient interface {
	Lookup(ctx context.Context, in *GaoRequest, opts ...grpc.CallOption) (*GaoResponse, error)
}

type gaoServiceClient struct {
	cc *grpc.ClientConn
}

func NewGaoServiceClient(cc *grpc.ClientConn) GaoServiceClient {
	return &gaoServiceClient{cc}
}

func (c *gaoServiceClient) Lookup(ctx context.Context, in *GaoRequest, opts ...grpc.CallOption) (*GaoResponse, error) {
	out := new(GaoResponse)
	err := grpc.Invoke(ctx, "/proto.GaoService/Lookup", in, out, c.cc, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// Server API for GaoService service

type GaoServiceServer interface {
	Lookup(context.Context, *GaoRequest) (*GaoResponse, error)
}

func RegisterGaoServiceServer(s *grpc.Server, srv GaoServiceServer) {
	s.RegisterService(&_GaoService_serviceDesc, srv)
}

func _GaoService_Lookup_Handler(srv interface{}, ctx context.Context, codec grpc.Codec, buf []byte) (interface{}, error) {
	in := new(GaoRequest)
	if err := codec.Unmarshal(buf, in); err != nil {
		return nil, err
	}
	out, err := srv.(GaoServiceServer).Lookup(ctx, in)
	if err != nil {
		return nil, err
	}
	return out, nil
}

var _GaoService_serviceDesc = grpc.ServiceDesc{
	ServiceName: "proto.GaoService",
	HandlerType: (*GaoServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Lookup",
			Handler:    _GaoService_Lookup_Handler,
		},
	},
	Streams: []grpc.StreamDesc{},
}
