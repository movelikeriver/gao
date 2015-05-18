// The gRPC Gao client.
//
// Usage:
//   go run gao/client/gao-client.go --user_id=10

package main

import (
	"flag"
	"fmt"

	pb "github.com/gao/proto"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/grpclog"
)

var (
	serverAddr = flag.String("server_addr", "127.0.0.1:10000", "host:port")
	userId     = flag.Int("user_id", 0, "user-id")
)

func main() {
	flag.Parse()

	var opts []grpc.DialOption
	fmt.Println("dailing", *serverAddr)
	conn, err := grpc.Dial(*serverAddr, opts...)
	if err != nil {
		grpclog.Fatalf("fail to dail: %v", err)
	}
	defer conn.Close()
	client := pb.NewGaoServiceClient(conn)

	req := new(pb.GaoRequest)
	req.UserId = int32(*userId)
	resp, err := client.Lookup(context.Background(), req)
	if err != nil {
		grpclog.Fatal("%v.Lookup(_) = _, %v", client, err)
	}
	fmt.Println(resp.String())
}
