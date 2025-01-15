import torch
from model import DQN 
from flappy_bird import FlappyBird
from torchvision import transforms
from torchvision.transforms.functional import rgb_to_grayscale

def preprocess(image):
    img = rgb_to_grayscale(transforms.Resize((84, 84))(
        torch.from_numpy(image).permute(2, 1, 0)
    ))
    threshold = 1 
    binary_img = (img < threshold).float()
    return binary_img

def evaluate_model(num_games, checkpoint_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DQN().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    scores = []
    total_steps = 0
    
    for game in range(num_games):
        game_state = FlappyBird()
        image, _, _ = game_state.next_frame(0)
        
        image = preprocess(image[:game_state.screen_width, :int(game_state.base_y)]).to(device)
        state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
        
        terminal = False
        steps = 0
        score = 0
        q_values = []
        
        while not terminal:
            with torch.no_grad():
                prediction = model(state)[0]
                q_values.append(prediction.cpu())
                action = torch.argmax(prediction).item()
            
            next_image, reward, terminal = game_state.next_frame(action)
            
            if reward == 1: score += 1

            if not terminal:
                next_image = preprocess(next_image[:game_state.screen_width, :int(game_state.base_y)]).to(device)
                next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
                state = next_state
            
            steps += 1
            total_steps += 1
        
        scores.append(score)
        print(f"Game {game + 1}: Score = {score}")
        
    avg_score = sum(scores) / len(scores)
    print(f"Average Score over {num_games} games: {avg_score:.2f}")
    return avg_score

if __name__ == "__main__":
    evaluate_model(num_games=150, checkpoint_path="logs/checkpoint_400000_f.pt")