// The gRPC Gao server.
//
// Usage:
//   go run gao/server/gao-server.go

package main

import (
	"flag"
	"fmt"
	"net"

	pb "github.com/gao/proto"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/grpclog"
)

var (
	port = flag.Int("port", 10000, "The server port")
)

type Profile struct {
	position string
	name     string
}

type gaoServer struct {
	names map[int32]Profile
}

func (s *gaoServer) init() {
	s.names = map[int32]Profile{
		20: {"F", "Sergio Agüero"},
		19: {"M", "Ricardo Álvarez"},
		21: {"GK", "Mariano Andújar"},
		23: {"D", "José María Basanta"},
		6: {"M", "Lucas Biglia"},
		3: {"D", "Hugo Campagnaro"},
		15: {"D", "Martin Demichelis"},
		7: {"M", "Ángel di María"},
		17: {"D", "Federico Fernández"},
		13: {"M", "Augusto Fernández"},
		5: {"M", "Fernando Gago"},
		2: {"D", "Ezequiel Garay"},
		9: {"F", "Gonzalo Higuaín"},
		22: {"F", "Ezequiel Lavezzi"},
		14: {"D/M", "Javier Mascherano"},
		10: {"F", "Lionel Messi"},
		12: {"GK", "Agustín Orión"},
		18: {"F", "Rodrigo Palacio"},
		8: {"M", "Enzo Perez"},
		11: {"M", "Maxi Rodríguez"},
		16: {"D", "Marcos Rojo"},
		1: {"GK", "Sergio Romero"},
		4: {"D", "Pablo Zabaleta"},
	}
}

func (s *gaoServer) Lookup(ctx context.Context, req *pb.GaoRequest) (*pb.GaoResponse, error) {
	resp := new(pb.GaoResponse)
	if profile, ok := s.names[req.UserId]; ok {
		resp.UserName = profile.name
		resp.Position = profile.position
	} else {
		resp.UserName = "no this name"
	}
	return resp, nil
}

func newServer() *gaoServer {
	s := new(gaoServer)
	s.init()
	return s
}

func main() {
	flag.Parse()

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		grpclog.Fatalf("failed listen: %v", err)
	}
	var opts []grpc.ServerOption
	grpcServer := grpc.NewServer(opts...)
	pb.RegisterGaoServiceServer(grpcServer, newServer())
	fmt.Println("starting server in port ", *port)
	grpcServer.Serve(lis)
}
