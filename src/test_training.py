import torch
from loss import *
from evaluation import evaluate_model
import time


def train(model, lrate, num_epoch, train_loader, test_loader, model_directory, early_stop_range, objective_rates, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    test_loss_per_epoch = []; train_average_loss_per_epoch = []; prediction_loss_per_epoch = []
    r2_list, r4_list, n2_list, n4_list = [], [], [], []


    print("="*50)
    print("SYSTEM INFO")
    print("="*50)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Batch Size: {48}")
    print(f"Num Workers: {train_loader.num_workers}")

    # Quick benchmark
    print("\n" + "="*50)
    print("BOTTLENECK TEST")
    print("="*50)

    # Test 1: Data loading speed
    start = time.time()
    for i, batch in enumerate(train_loader):
        if i >= 20:
            break
    data_time = time.time() - start
    print(f"20 batches data loading: {data_time:.2f}s ({data_time/20:.3f}s per batch)")

    # Test 2: GPU transfer speed
    batch = next(iter(train_loader))
    start = time.time()
    for _ in range(20):
        batch[0].to(device)
        batch[1].to(device)
    transfer_time = time.time() - start
    print(f"20 CPU->GPU transfers: {transfer_time:.2f}s ({transfer_time/20:.3f}s per batch)")

    # Test 3: Forward pass speed
    model.eval()
    batch = next(iter(train_loader))
    patients = batch[0].to(device)
    visit_lens = batch[3]
    start = time.time()
    with torch.no_grad():
        for _ in range(20):
            pred, tp_list, recon_h_list, alphas = model(patients, visit_lens)
    forward_time = time.time() - start
    print(f"20 forward passes: {forward_time:.2f}s ({forward_time/20:.3f}s per batch)")

    # Test 4: Full training iteration
    model.train()
    batch = next(iter(train_loader))
    patients = batch[0].to(device)
    labels = batch[1].to(device)
    visit_lens = batch[3]

    start = time.time()
    for _ in range(20):
        pred, tp_list, recon_h_list, alphas = model(patients, visit_lens)
        loss, _, _ = shy_loss(pred, labels.to(torch.float32), patients, recon_h_list, 
                            tp_list, alphas, visit_lens.tolist(), objective_rates, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    full_time = time.time() - start
    print(f"20 full iterations: {full_time:.2f}s ({full_time/20:.3f}s per batch)")

    print("\n" + "="*50)
    print("BOTTLENECK ANALYSIS")
    print("="*50)
    if data_time/20 > 0.5:
        print("⚠️  DATA LOADING IS SLOW - Increase num_workers and batch_size")
    if transfer_time/20 > 0.1:
        print("⚠️  CPU->GPU TRANSFER IS SLOW - Use pin_memory and non_blocking")
    if forward_time/20 > 1.0:
        print("⚠️  FORWARD PASS IS SLOW - Consider model simplification")
    if full_time/20 < 1.0:
        print("✓ Training speed is reasonable")
    else:
        print(f"⚠️  Full iteration takes {full_time/20:.2f}s - should be <1s")

    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)
    time_ratio = data_time / full_time
    if time_ratio > 0.3:
        print(f"1. Data loading is {time_ratio*100:.0f}% of time - FIX THIS FIRST")
        print("   - Increase num_workers to 8")
        print("   - Increase batch_size to 128-256")
        print("   - Preload data into RAM if using MIMIC_IV")
        print("\n2. GPU upgrade won't help much until data loading is fixed")
    else:
        print("1. Data loading is optimized")
        print(f"2. GPU utilization seems {'good' if full_time/20 < 1.0 else 'poor'}")
        if full_time/20 > 1.0:
            print("   - A faster GPU might help (2-3x speedup possible)")
            print("   - But also try: mixed precision, torch.compile, larger batches")


    for epoch in range(num_epoch):
        one_epoch_train_loss = []
        print(f"Number of batches in train_loader: {len(train_loader)}")
        for i, (patients, labels, pids, visit_lens) in enumerate(train_loader):
            if i % 10 == 0:
                print(f"Processing batch {i+1}/{len(train_loader)}")
            patients = patients.to(device); labels = labels.to(device)
            #print("Moved")
            pred, tp_list, recon_h_list, alphas = model(patients, visit_lens)
            #print("Got Prediction")
            loss, _, _ = shy_loss(pred, labels.to(torch.float32), patients, recon_h_list, tp_list, alphas, visit_lens.tolist(), objective_rates, device)
            ##print("Got Loss")
            one_epoch_train_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_average_loss_per_epoch.append((sum(one_epoch_train_loss) / len(one_epoch_train_loss)).item())
        print('Epoch: [{}/{}], Training Loss: {:.9f}'.format(epoch+1, num_epoch, train_average_loss_per_epoch[-1]))
        model.eval()
        pred_list = []; label_list = []; original_patient_list = []; test_tp_list = []; test_recon_list = []; visit_lens_list = []; alpha_list = []
        for (test_patients, test_labels, test_pids, test_visit_lens) in test_loader:
            test_patients = test_patients.to(device); test_labels = test_labels.to(device)
            with torch.no_grad():
              pred, tp_list, recon_h_list, alphas = model(test_patients, test_visit_lens)
            pred_list.append(pred); label_list.append(test_labels); original_patient_list.append(test_patients); alpha_list.append(alphas)
            test_tp_list += tp_list; test_recon_list += recon_h_list; visit_lens_list += test_visit_lens.tolist()
        pred = torch.vstack(pred_list); test_labels = torch.vstack(label_list); test_patients = torch.vstack(original_patient_list); test_alphas = torch.vstack(alpha_list)
        test_loss, loss_list, name_list = shy_loss(pred, test_labels.to(torch.float32), test_patients, test_recon_list, test_tp_list, test_alphas, visit_lens_list, objective_rates, device)
        test_loss_per_epoch.append(test_loss.item())
        prediction_loss_per_epoch.append(loss_list[0].item())
        _, _, _, _, metric_r2, metric_n2, _, _, _, _, metric_r4, metric_n4, _, _, _, _, _, _, = evaluate_model(pred, test_labels, 5, 10, 15, 20, 25, 30)
        r2_list.append(metric_r2); r4_list.append(metric_r4); n2_list.append(metric_n2); n4_list.append(metric_n4)
        print('Test Epoch {}: {:.9f} (recall@10); {:.9f} (recall@20); {:.9f} (ndcg@10); {:.9f} (ndcg@20)'.format(epoch+1, metric_r2, metric_r4, metric_n2, metric_n4))
        if len(test_tp_list[0].shape) > 2:
            print('{}: {:.9f}; {}: {:.9f}; {}: {:.9f}; {}: {:.9f}'.format(name_list[0], loss_list[0].item(), name_list[1], loss_list[1], name_list[2], loss_list[2].item(), name_list[3], loss_list[3].item()))
        else:
            print('{}: {:.9f}; {}: {:.9f}'.format(name_list[0], loss_list[0].item(), name_list[1], loss_list[1]))
        if epoch >= 30 and prediction_loss_per_epoch[-1] < min(prediction_loss_per_epoch[0:-1]):
           torch.save(model.state_dict(), f'../saved_models/{model_directory}/shy_epoch_{epoch+1}.pth')
        early_stop = (-1) * early_stop_range
        last_loss = prediction_loss_per_epoch[early_stop:]
        if epoch >= 30 and sorted(last_loss) == last_loss:
           break
        model.train()
    return r2_list, r4_list, n2_list, n4_list, test_loss_per_epoch, train_average_loss_per_epoch, prediction_loss_per_epoch

